import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Thẩm định hiệu quả dự án đầu tư", page_icon="📈", layout="wide")

# ======================================================================
# CÁC HÀM TÍNH TOÁN
# ======================================================================


def npv(rate, cfs):
    """NPV = sum( CF_t / (1+r)^t ),  t = 0..n"""
    t = np.arange(len(cfs))
    return float(np.sum(np.asarray(cfs, dtype=float) / (1 + rate) ** t))


def irr(cfs):
    """Tìm các nghiệm r sao cho NPV(r) = 0 (quét lưới + chia đôi)."""
    cfs = np.asarray(cfs, dtype=float)
    if not (np.any(cfs < 0) and np.any(cfs > 0)):
        return []
    t = np.arange(len(cfs))
    f = lambda r: float(np.sum(cfs / (1 + r) ** t))

    grid = np.linspace(-0.99, 10, 4000)
    vals = np.array([f(r) for r in grid])
    roots = []
    for i in range(len(grid) - 1):
        a, b, fa, fb = grid[i], grid[i + 1], vals[i], vals[i + 1]
        if fa == 0:
            roots.append(a)
        elif fa * fb < 0:
            for _ in range(100):
                m = (a + b) / 2
                fm = f(m)
                if fa * fm <= 0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm
            roots.append((a + b) / 2)
    return roots


def mirr(cfs, finance_rate, reinvest_rate):
    """MIRR = (FV dòng tiền dương tại lãi suất tái đầu tư / |PV dòng tiền âm tại lãi suất tài trợ|)^(1/n) - 1"""
    cfs = np.asarray(cfs, dtype=float)
    n = len(cfs) - 1
    t = np.arange(len(cfs))
    pos = np.where(cfs > 0, cfs, 0.0)
    neg = np.where(cfs < 0, cfs, 0.0)
    fv_pos = float(np.sum(pos * (1 + reinvest_rate) ** (n - t)))
    pv_neg = float(np.sum(neg / (1 + finance_rate) ** t))
    if fv_pos <= 0 or pv_neg == 0 or n <= 0:
        return None, fv_pos, pv_neg
    return (fv_pos / abs(pv_neg)) ** (1 / n) - 1, fv_pos, pv_neg


def bcr(rate, benefits, costs):
    """BCR = PV(Lợi ích) / PV(Chi phí)"""
    t = np.arange(len(benefits))
    pv_b = float(np.sum(np.asarray(benefits, dtype=float) / (1 + rate) ** t))
    pv_c = float(np.sum(np.asarray(costs, dtype=float) / (1 + rate) ** t))
    return (pv_b / pv_c if pv_c != 0 else None), pv_b, pv_c


def payback(cfs):
    """Thời gian hoàn vốn (nội suy tuyến tính trong năm hoàn vốn)."""
    cum = np.cumsum(cfs)
    if cum[0] >= 0:
        return 0.0, cum
    for i in range(1, len(cfs)):
        if cum[i] >= 0:
            if cfs[i] == 0:
                return float(i), cum
            return (i - 1) + (-cum[i - 1]) / cfs[i], cum
    return None, cum


def fmt(x, d=2):
    return f"{x:,.{d}f}"


# ======================================================================
# GIAO DIỆN
# ======================================================================
st.title("📈 Thẩm định hiệu quả dự án đầu tư")
st.caption("Chọn chỉ tiêu cần tính, sau đó nhập dữ liệu dự án.")

METRICS = {
    "NPV – Giá trị hiện tại ròng": "NPV",
    "IRR – Tỷ suất hoàn vốn nội bộ": "IRR",
    "MIRR – IRR hiệu chỉnh": "MIRR",
    "BCR – Tỷ số lợi ích / chi phí": "BCR",
    "PP – Thời gian hoàn vốn": "PP",
    "DPP – Thời gian hoàn vốn có chiết khấu": "DPP",
}

choice = st.selectbox("❓ Bạn muốn tính chỉ tiêu nào?", list(METRICS.keys()), index=None,
                      placeholder="Chọn một chỉ tiêu...")

if choice is None:
    st.info("👆 Vui lòng chọn chỉ tiêu cần tính để bắt đầu.")
    st.stop()

m = METRICS[choice]

# ---------------- Công thức ----------------
with st.expander("📐 Công thức", expanded=True):
    if m == "NPV":
        st.latex(r"NPV=\sum_{t=0}^{n}\frac{CF_t}{(1+r)^t}")
        st.write("Quy tắc: **NPV > 0** → dự án chấp nhận được.")
    elif m == "IRR":
        st.latex(r"\sum_{t=0}^{n}\frac{CF_t}{(1+IRR)^t}=0")
        st.write("Quy tắc: **IRR > r (suất chiết khấu / chi phí vốn)** → dự án chấp nhận được.")
    elif m == "MIRR":
        st.latex(r"MIRR=\left(\frac{\sum_{t=0}^{n} CF^{+}_t(1+r_{tđ})^{n-t}}"
                 r"{\left|\sum_{t=0}^{n}\frac{CF^{-}_t}{(1+r_{tt})^{t}}\right|}\right)^{1/n}-1")
        st.write("Trong đó $r_{tđ}$: lãi suất tái đầu tư; $r_{tt}$: lãi suất tài trợ (chi phí vốn vay). "
                 "Quy tắc: **MIRR > r** → dự án chấp nhận được.")
    elif m == "BCR":
        st.latex(r"BCR=\frac{PV(B)}{PV(C)}=\frac{\sum_{t=0}^{n}\frac{B_t}{(1+r)^t}}{\sum_{t=0}^{n}\frac{C_t}{(1+r)^t}}")
        st.write("Quy tắc: **BCR > 1** → dự án chấp nhận được.")
    elif m == "PP":
        st.latex(r"PP=(k-1)+\frac{\left|\sum_{t=0}^{k-1}CF_t\right|}{CF_k}")
        st.write("Với $k$ là năm đầu tiên mà dòng tiền tích lũy ≥ 0. Thời gian hoàn vốn càng ngắn càng tốt.")
    elif m == "DPP":
        st.latex(r"DPP=(k-1)+\frac{\left|\sum_{t=0}^{k-1}\frac{CF_t}{(1+r)^t}\right|}{\frac{CF_k}{(1+r)^k}}")
        st.write("Với $k$ là năm đầu tiên mà dòng tiền chiết khấu tích lũy ≥ 0.")

st.divider()
st.subheader("📝 Nhập dữ liệu dự án")

col1, col2 = st.columns(2)
with col1:
    n = st.number_input("Số năm hoạt động của dự án (n)", min_value=1, max_value=50, value=5, step=1)

# Tham số lãi suất
r = fin = reinv = None
if m in ("NPV", "BCR", "DPP"):
    with col2:
        r = st.number_input("Suất chiết khấu r (%/năm)", value=10.0, step=0.5, format="%.2f") / 100
elif m == "IRR":
    with col2:
        r = st.number_input("Suất chiết khấu / chi phí vốn r (%/năm) – để so sánh với IRR",
                            value=10.0, step=0.5, format="%.2f") / 100
elif m == "MIRR":
    with col2:
        fin = st.number_input("Lãi suất tài trợ (%/năm)", value=8.0, step=0.5, format="%.2f") / 100
        reinv = st.number_input("Lãi suất tái đầu tư (%/năm)", value=10.0, step=0.5, format="%.2f") / 100
        r = st.number_input("Chi phí vốn r để so sánh (%/năm)", value=10.0, step=0.5, format="%.2f") / 100

years = list(range(0, int(n) + 1))

if m == "BCR":
    st.write("Nhập **Lợi ích (B)** và **Chi phí (C)** của từng năm (năm 0 thường là vốn đầu tư ban đầu):")
    df0 = pd.DataFrame({"Năm": years, "Lợi ích B": 0.0, "Chi phí C": 0.0})
    df_in = st.data_editor(df0, hide_index=True, disabled=["Năm"], use_container_width=True,
                           key=f"bcr_{n}")
else:
    st.write("Nhập **dòng tiền ròng CF** từng năm (năm 0 là vốn đầu tư, nhập **số âm**):")
    default = [0.0] * (int(n) + 1)
    df0 = pd.DataFrame({"Năm": years, "Dòng tiền ròng CF": default})
    df_in = st.data_editor(df0, hide_index=True, disabled=["Năm"], use_container_width=True,
                           key=f"cf_{m}_{n}")

calc = st.button("🧮 Tính toán", type="primary")

# ======================================================================
# TÍNH TOÁN & HIỂN THỊ KẾT QUẢ
# ======================================================================
if calc:
    st.divider()
    st.subheader("✅ Kết quả")

    if m == "BCR":
        B = df_in["Lợi ích B"].fillna(0).to_numpy(dtype=float)
        C = df_in["Chi phí C"].fillna(0).to_numpy(dtype=float)
        ratio, pv_b, pv_c = bcr(r, B, C)
        a, b, c = st.columns(3)
        a.metric("PV lợi ích", fmt(pv_b))
        b.metric("PV chi phí", fmt(pv_c))
        if ratio is None:
            st.error("Tổng chi phí hiện tại bằng 0, không thể tính BCR.")
        else:
            c.metric("BCR", fmt(ratio, 4))
            (st.success if ratio > 1 else st.warning)(
                "BCR > 1 → Nên chấp nhận dự án." if ratio > 1 else "BCR ≤ 1 → Không nên chấp nhận dự án.")
        t = np.arange(len(B))
        st.dataframe(pd.DataFrame({"Năm": t, "B": B, "C": C,
                                   "PV(B)": B / (1 + r) ** t, "PV(C)": C / (1 + r) ** t}),
                     hide_index=True, use_container_width=True)

    else:
        cfs = df_in["Dòng tiền ròng CF"].fillna(0).to_numpy(dtype=float)
        t = np.arange(len(cfs))

        if not (np.any(cfs < 0) and np.any(cfs > 0)) and m != "NPV":
            st.warning("Dòng tiền cần có cả giá trị âm (đầu tư) và dương (thu nhập) để tính chỉ tiêu này.")

        if m == "NPV":
            pv = cfs / (1 + r) ** t
            val = float(pv.sum())
            st.metric("NPV", fmt(val))
            (st.success if val > 0 else st.warning)(
                "NPV > 0 → Nên chấp nhận dự án." if val > 0 else "NPV ≤ 0 → Không nên chấp nhận dự án.")
            st.dataframe(pd.DataFrame({"Năm": t, "CF": cfs, "Hệ số chiết khấu": 1 / (1 + r) ** t,
                                       "PV(CF)": pv, "PV tích lũy": np.cumsum(pv)}),
                         hide_index=True, use_container_width=True)

        elif m == "IRR":
            roots = irr(cfs)
            if not roots:
                st.error("Không tìm được IRR (dòng tiền phải đổi dấu ít nhất một lần).")
            else:
                val = roots[0]
                st.metric("IRR", f"{val * 100:.4f} %")
                if len(roots) > 1:
                    st.warning("Dòng tiền đổi dấu nhiều lần nên có nhiều nghiệm IRR: "
                               + ", ".join(f"{x * 100:.2f}%" for x in roots)
                               + ". Nên dùng NPV hoặc MIRR để ra quyết định.")
                (st.success if val > r else st.warning)(
                    f"IRR ({val * 100:.2f}%) > r ({r * 100:.2f}%) → Nên chấp nhận dự án." if val > r
                    else f"IRR ({val * 100:.2f}%) ≤ r ({r * 100:.2f}%) → Không nên chấp nhận dự án.")

        elif m == "MIRR":
            val, fv_pos, pv_neg = mirr(cfs, fin, reinv)
            a, b = st.columns(2)
            a.metric("FV dòng tiền dương (tại lãi suất tái đầu tư)", fmt(fv_pos))
            b.metric("PV dòng tiền âm (tại lãi suất tài trợ)", fmt(pv_neg))
            if val is None:
                st.error("Không tính được MIRR (cần có dòng tiền dương và dòng tiền âm).")
            else:
                st.metric("MIRR", f"{val * 100:.4f} %")
                (st.success if val > r else st.warning)(
                    f"MIRR ({val * 100:.2f}%) > r ({r * 100:.2f}%) → Nên chấp nhận dự án." if val > r
                    else f"MIRR ({val * 100:.2f}%) ≤ r ({r * 100:.2f}%) → Không nên chấp nhận dự án.")

        elif m == "PP":
            val, cum = payback(cfs)
            if val is None:
                st.error("Dự án chưa hoàn vốn trong thời gian phân tích.")
            else:
                st.metric("Thời gian hoàn vốn (PP)", f"{val:.2f} năm")
                st.caption(f"≈ {int(val)} năm {round((val - int(val)) * 12)} tháng")
            st.dataframe(pd.DataFrame({"Năm": t, "CF": cfs, "CF tích lũy": cum}),
                         hide_index=True, use_container_width=True)
            st.line_chart(pd.DataFrame({"CF tích lũy": cum}, index=t))

        elif m == "DPP":
            pv = cfs / (1 + r) ** t
            val, cum = payback(pv)
            if val is None:
                st.error("Dự án chưa hoàn vốn (có chiết khấu) trong thời gian phân tích.")
            else:
                st.metric("Thời gian hoàn vốn có chiết khấu (DPP)", f"{val:.2f} năm")
                st.caption(f"≈ {int(val)} năm {round((val - int(val)) * 12)} tháng")
            st.dataframe(pd.DataFrame({"Năm": t, "CF": cfs, "PV(CF)": pv, "PV tích lũy": cum}),
                         hide_index=True, use_container_width=True)
            st.line_chart(pd.DataFrame({"PV tích lũy": cum}, index=t))
