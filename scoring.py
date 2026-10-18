"""
TrustGate scoring engine: features -> rule flags -> XGBoost probability -> trust score -> decision -> reasons.

Design for portability (IBM LinuxONE = s390x, where some wheels are missing):
  * Feature extraction, rules and model inference are PURE PYTHON (standard library only).
  * The XGBoost model is read from its JSON file and evaluated by a tiny tree walker (PureXGB).
  * If the `xgboost` package happens to be importable, it is used ONLY to compute SHAP contributions
    for the "reasons"; otherwise reasons come from rule flags + unusual feature values.
The feature code mirrors the Colab notebook (Step 2) line by line, so scores match the notebook.
"""
import json
import math
import struct
from pathlib import Path

NAN = float("nan")
REQ = {
    "purchase_order": ["doc_number", "date", "buyer", "seller"],
    "invoice": ["doc_number", "date", "seller", "buyer", "total_amount"],
    "packing_list": ["doc_number", "gross_weight_kg", "total_packages"],
    "bill_of_lading": ["doc_number", "consignee", "port_of_loading", "ship_on_board_date", "container_no"],
}
DOC_KEYS = list(REQ)


def isnan(x):
    return x != x


def is_empty(x):
    return x is None or x == ""


def num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else NAN


def parse_date(x):
    from datetime import date
    try:
        return date.fromisoformat(str(x)[:10])
    except Exception:
        return None


def days(a, b):
    return NAN if (a is None or b is None) else float((a - b).days)


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


# ------------------------------------------------------------------ history (what the system already knows)
class History:
    """Seller / invoice history. In production this is a database query over previously recorded shipments."""

    def __init__(self, data=None):
        data = data or {}
        self.inv = dict(data.get("inv", {}))
        self.n = dict(data.get("n", {}))
        self.sum = dict(data.get("sum", {}))
        self.qsum = dict(data.get("qsum", {}))

    def update(self, ship):
        inv = ship["documents"]["invoice"]
        no, seller = inv.get("doc_number"), inv.get("seller")
        ta = num(inv.get("total_amount"))
        qty_total = sum(i["qty"] for i in inv["items"])
        if not is_empty(no):
            self.inv[no] = self.inv.get(no, 0) + 1
        self.n[seller] = self.n.get(seller, 0) + 1
        if not isnan(ta):
            self.sum[seller] = self.sum.get(seller, 0.0) + ta
        self.qsum[seller] = self.qsum.get(seller, 0) + qty_total


# ------------------------------------------------------------------ features (same as notebook Step 2)
def extract_features(ship, hist, ref):
    prod, sellers, valid_hs = ref["products"], set(ref["registered_sellers"]), set(ref["valid_hs_codes"])
    d = ship["documents"]
    po, inv, pl, bol = d["purchase_order"], d["invoice"], d["packing_list"], d["bill_of_lading"]
    f = {}

    f["n_missing"] = sum(is_empty(d[k].get(x)) for k, fields in REQ.items() for x in fields)

    f["line_err_rel"] = max(abs(i["qty"] * i["unit_price"] - i["total"]) / max(i["total"], 1) for i in inv["items"])
    ta = num(inv.get("total_amount"))
    f["grand_err_rel"] = (abs(sum(i["total"] for i in inv["items"]) - ta) / max(ta, 1)) if not isnan(ta) else NAN
    po_d, inv_d = parse_date(po.get("date")), parse_date(inv.get("date"))
    ship_d, eta_d = parse_date(bol.get("ship_on_board_date")), parse_date(bol.get("delivery_eta"))
    f["eta_minus_ship"] = days(eta_d, ship_d)
    f["inv_minus_ship"] = days(inv_d, ship_d)
    f["po_minus_inv"] = days(po_d, inv_d)

    f["qty_mismatch_pl"] = sum(a["qty"] != b["qty"] for a, b in zip(inv["items"], pl["items"]))
    f["qty_mismatch_po"] = sum(a["qty"] != c["qty"] for a, c in zip(inv["items"], po["items"]))
    cons, buyer = bol.get("consignee"), inv.get("buyer")
    f["consignee_match"] = NAN if (is_empty(cons) or is_empty(buyer)) else float(cons == buyer)
    wb, wp = num(bol.get("gross_weight_kg")), num(pl.get("gross_weight_kg"))
    f["weight_diff_rel"] = abs(wb - wp) / wp if not (isnan(wb) or isnan(wp) or wp == 0) else NAN
    exp_w = sum(i["qty"] * prod[i["product_id"]]["unit_weight_kg"] for i in pl["items"] if i.get("product_id") in prod)
    f["weight_ratio_expected"] = wp / exp_w if (exp_w and not isnan(wp)) else NAN

    no, seller = inv.get("doc_number"), inv.get("seller")
    f["invoice_seen_before"] = 0 if is_empty(no) else hist.inv.get(no, 0)
    f["seller_known"] = 0.0 if is_empty(seller) else float(seller in sellers)
    f["seller_prev_count"] = hist.n.get(seller, 0)
    prev_mean = hist.sum.get(seller, 0.0) / hist.n[seller] if hist.n.get(seller) else NAN
    f["amount_vs_seller_mean"] = ta / prev_mean if not (isnan(ta) or isnan(prev_mean) or prev_mean == 0) else NAN
    qty_total = sum(i["qty"] for i in inv["items"])
    f["qty_total"] = qty_total
    prev_q = hist.qsum.get(seller, 0) / hist.n[seller] if hist.n.get(seller) else NAN
    f["qty_vs_seller_mean"] = qty_total / prev_q if not (isnan(prev_q) or prev_q == 0) else NAN

    ratios = [i["unit_price"] / prod[i["product_id"]]["list_price"]
              for i in inv["items"] if i.get("product_id") in prod and prod[i["product_id"]]["list_price"]]
    f["price_ratio_min"] = min(ratios) if ratios else NAN
    f["price_ratio_max"] = max(ratios) if ratios else NAN
    f["hs_valid_all"] = float(all(i["hs_code"] in valid_hs for i in inv["items"]))
    f["n_items"] = len(inv["items"])
    f["log_amount"] = math.log1p(ta) if not isnan(ta) else NAN
    return f


def gt(a, b):
    return (not isnan(a)) and a > b


def lt(a, b):
    return (not isnan(a)) and a < b


def rule_flags(f):
    return {
        "r_missing": int(f["n_missing"] > 0),
        "r_calc": int(gt(f["line_err_rel"], 1e-3) or gt(f["grand_err_rel"], 1e-3)),
        "r_date": int((not isnan(f["eta_minus_ship"]) and f["eta_minus_ship"] <= 0)
                      or gt(f["inv_minus_ship"], 0) or gt(f["po_minus_inv"], 0)),
        "r_cross": int(f["qty_mismatch_pl"] > 0 or f["qty_mismatch_po"] > 0 or f["consignee_match"] == 0
                       or gt(f["weight_diff_rel"], 1e-3)),
        "r_dup": int(f["invoice_seen_before"] > 0),
        "r_seller_hs": int(f["seller_known"] == 0 or f["hs_valid_all"] == 0),
        "r_price": int(lt(f["price_ratio_min"], 0.375) or gt(f["price_ratio_max"], 2)),
    }


# ------------------------------------------------------------------ XGBoost JSON model, evaluated in pure Python
class PureXGB:
    def __init__(self, path, n_trees=None):
        m = json.load(open(path, encoding="utf-8"))
        learner = m["learner"]
        bs = str(learner["learner_model_param"]["base_score"]).strip("[]")
        base = float(bs)
        self.base_margin = math.log(base / (1 - base))
        self.best_iteration = int(learner.get("attributes", {}).get("best_iteration", -1))
        trees = learner["gradient_booster"]["model"]["trees"]
        if n_trees is None:
            n_trees = self.best_iteration + 1 if self.best_iteration >= 0 else len(trees)
        self.n_trees = min(n_trees, len(trees))
        self.trees = [(t["left_children"], t["right_children"], t["split_indices"],
                       [f32(c) for c in t["split_conditions"]], t["default_left"]) for t in trees[: self.n_trees]]

    def margin(self, x):
        x = [v if isnan(v) else f32(v) for v in x]
        total = self.base_margin
        for left, right, idx, cond, dleft in self.trees:
            n = 0
            while left[n] != -1:
                v = x[idx[n]]
                if isnan(v):
                    n = left[n] if dleft[n] else right[n]
                else:
                    n = left[n] if v < cond[n] else right[n]
            total += cond[n]
        return total

    def predict(self, x):
        m = self.margin(x)
        return 1.0 / (1.0 + math.exp(-m))


# ------------------------------------------------------------------ the scorer
class Scorer:
    def __init__(self, model_dir):
        md = Path(model_dir)
        self.meta = json.load(open(md / "model_meta.json", encoding="utf-8"))
        self.ref = json.load(open(md / "reference_data.json", encoding="utf-8"))
        self.clean_ranges = json.load(open(md / "clean_ranges.json", encoding="utf-8"))
        self.features = self.meta["features"]
        self.reason_text = dict(self.meta["reason_text"])
        self.reason_text.update({                       # wording tuned for the console (values 1.0 to 1.3x list are legit premiums)
            "price_ratio_max": "a unit price is above the usual range ({v:.2f}x the list price)",
            "price_ratio_min": "a unit price is below the usual range ({v:.2f}x the list price)",
        })
        self.approve_min = self.meta["decision_thresholds"]["approve_min"]
        self.review_min = self.meta["decision_thresholds"]["review_min"]
        self.cap = self.meta["rule_cap_score"]
        n_trees = self.meta.get("best_iteration")
        self.model = PureXGB(md / "trustgate_xgb.json", None if n_trees is None else n_trees + 1)
        self.history_path = md / "history.json"
        self.booster = None
        try:                                    # optional: SHAP via xgboost if it is installed
            import xgboost as xgb
            self.booster = xgb.Booster()
            self.booster.load_model(str(md / "trustgate_xgb.json"))
            self._xgb = xgb
        except Exception:
            self.booster = None

    @property
    def engine(self):
        return "pure-python model + xgboost SHAP" if self.booster else "pure-python model (xgboost not installed)"

    def fresh_history(self):
        data = json.load(open(self.history_path, encoding="utf-8")) if self.history_path.exists() else {}
        return History(data)

    def _text(self, feat, value):
        t = self.reason_text.get(feat, feat)
        try:
            return t.format(v=value)
        except Exception:
            return t

    def _unusual(self, feat, v):
        lo, hi = self.clean_ranges[feat]
        return (not isnan(v)) and (v < lo or v > hi)

    def _reasons(self, vec, flags, k=5):
        feats = self.features
        if self.booster is not None:
            try:
                dm = self._xgb.DMatrix([vec], feature_names=feats)
                contrib = self.booster.predict(dm, pred_contribs=True, iteration_range=(0, self.model.n_trees))[0]
                order = sorted(range(len(feats)), key=lambda i: -contrib[i])
                out = []
                for i in order:
                    if contrib[i] <= 0.05:
                        break
                    if self._unusual(feats[i], vec[i]):
                        out.append(self._text(feats[i], vec[i]))
                    if len(out) == k:
                        break
                if out:
                    return out
            except Exception:
                pass
        # fallback without SHAP: fired rules first, then feature values furthest outside the clean range
        out = [self._text(r, 1) for r, on in flags.items() if on]
        scored = []
        for i, name in enumerate(feats):
            if name.startswith("r_") or not self._unusual(name, vec[i]):
                continue
            lo, hi = self.clean_ranges[name]
            dist = (lo - vec[i]) if vec[i] < lo else (vec[i] - hi)
            scored.append((dist / max(hi - lo, abs(hi), 1e-9), name, vec[i]))
        for _, name, v in sorted(scored, reverse=True):
            out.append(self._text(name, v))
        return out[:k] or ["no single dominant reason (many small signals add up)"]

    def score(self, ship, hist):
        f = extract_features(ship, hist, self.ref)
        flags = rule_flags(f)
        row = {**f, **flags}
        vec = [float(row[name]) for name in self.features]
        p = self.model.predict(vec)
        ml_trust = int(round(100 * (1 - p)))
        any_rule = any(flags.values())
        trust = min(ml_trust, self.cap) if any_rule else ml_trust
        decision = "approve" if trust >= self.approve_min else ("review" if trust >= self.review_min else "reject")
        reasons = [r for r in self._reasons(vec, flags) if not r.startswith("rule: ")]   # rule failures are listed separately
        if not reasons or reasons == ["no single dominant reason (many small signals add up)"]:
            reasons = (["See the failed rule checks"] if any_rule else
                       (["All rule checks passed and every value is inside the range seen in clean shipments"]
                        if decision == "approve" else ["No single dominant reason (many small signals add up)"]))
        return {
            "trust_score": trust,
            "ml_trust_score": ml_trust,
            "fraud_probability": round(p, 4),
            "rule_cap_applied": bool(any_rule and ml_trust > self.cap),
            "decision": decision,
            "rule_flags": [{"rule": r, "message": self.reason_text.get(r, r).replace("rule: ", "")}
                           for r, on in flags.items() if on],
            "reasons": reasons,
            "features": {k: (None if isnan(v) else round(v, 4)) for k, v in f.items()},
            "engine": self.engine,
        }
