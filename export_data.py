import sqlite3
import csv
import json
from pathlib import Path
from fpdf import FPDF, errors as fpdf_errors

# -------------------------
# ĐƯỜNG DẪN CƠ BẢN
# -------------------------

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "wellbeing_logs.db"
EXPORT_DIR = BASE_DIR / "exports"
CSV_PATH = EXPORT_DIR / "wellbeing_conversations.csv"
REPORTS_DIR = EXPORT_DIR / "reports"


# Font Unicode (đúng với cấu trúc hiện tại của bạn)
FONT_PATH = BASE_DIR / "fonts" / "dejavu-sans" / "DejaVuSans.ttf"


# -------------------------
# ĐỌC DỮ LIỆU TỪ SQLITE
# -------------------------

def load_messages():
    """Đọc toàn bộ bản ghi từ bảng messages trong SQLite."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            id,
            user_id,
            condition,
            lang_code,
            user_message,
            assistant_reply,
            emotion_json,
            safety_json,
            supervisor_json
        FROM messages
        ORDER BY id ASC
        """
    )

    rows = cur.fetchall()
    conn.close()
    return rows


# -------------------------
# EXPORT CSV
# -------------------------

def export_csv(messages):
    """
    Xuất toàn bộ hội thoại ra 1 file CSV cho mục đích nghiên cứu.
    Mỗi dòng = 1 lượt (turn).
    """
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    header = [
        "turn_id",
        "user_id",
        "condition",
        "language",
        "user_text",
        "agent_text",
        "emotion",
        "stress_level",
        "main_issue",
        "safety_flag",
        "safety_notes",
    ]

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for row in messages:
            emotion = json.loads(row["emotion_json"]) if row["emotion_json"] else {}
            safety = json.loads(row["safety_json"]) if row["safety_json"] else {}

            writer.writerow(
                [
                    row["id"],
                    row["user_id"],
                    row["condition"],
                    row["lang_code"],
                    row["user_message"],
                    row["assistant_reply"],
                    emotion.get("primary_emotion", ""),
                    emotion.get("stress_level", ""),
                    emotion.get("main_issue", ""),
                    1 if safety.get("is_risk") else 0,
                    safety.get("notes", ""),
                ]
            )

    print(f"[OK] CSV exported to: {CSV_PATH.resolve()}")


# -------------------------
# LỚP PDF TÓM TẮT
# -------------------------

class ReportPDF(FPDF):
    """PDF summary cho mỗi user_id."""

    def __init__(self, *args, **kwargs):
        # A4, mm cho chắc
        super().__init__(orientation="P", unit="mm", format="A4", *args, **kwargs)

        if not FONT_PATH.exists():
            raise FileNotFoundError(f"Không tìm thấy font Unicode: {FONT_PATH}")

        # Đăng ký font Unicode
        self.add_font("DejaVu", "", str(FONT_PATH), uni=True)

        # Thiết lập lề rõ ràng để tránh lỗi width
        self.set_margins(left=15, top=15, right=15)

    def header(self):
        self.set_font("DejaVu", "", 14)
        # width = 180mm (A4 ~210mm, trừ lề còn dư sức)
        self.multi_cell(180, 8, "Wellbeing Conversation Summary", align="C")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "", 8)
        self.multi_cell(180, 8, f"Page {self.page_no()}", align="C")


# -------------------------
# EXPORT PDF SUMMARY
# -------------------------

def export_pdf(messages):
    """
    Xuất PDF tóm tắt cho mỗi user_id:
    - Tổng số lượt
    - Ngôn ngữ
    - Điều kiện (condition)
    - Các cảm xúc xuất hiện
    - Số lượt bị flag risk
    (Không in chi tiết hội thoại để tránh lỗi text dài, CSV đã lưu đủ.)
    """

    reports_dir = EXPORT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Gom theo user_id
    sessions = {}
    for row in messages:
        sid = row["user_id"] or "unknown_user"
        sessions.setdefault(sid, []).append(row)

    for user_id, turns in sessions.items():
        # Thống kê cơ bản
        num_turns = len(turns)
        langs = sorted({t["lang_code"] for t in turns if t["lang_code"]})
        conditions = sorted({t["condition"] for t in turns if t["condition"]})

        emotions = []
        risk_count = 0

        for t in turns:
            if t["emotion_json"]:
                try:
                    e = json.loads(t["emotion_json"])
                    if e.get("primary_emotion"):
                        emotions.append(e["primary_emotion"])
                except Exception:
                    pass

            if t["safety_json"]:
                try:
                    s = json.loads(t["safety_json"])
                    if s.get("is_risk"):
                        risk_count += 1
                except Exception:
                    pass

        emotions = sorted(set(emotions))

        pdf = ReportPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        w = 180  # chiều rộng vùng viết nội dung

        pdf.set_font("DejaVu", "", 12)
        pdf.multi_cell(w, 6, f"User ID: {user_id}")
        pdf.multi_cell(w, 6, f"Total turns: {num_turns}")
        pdf.multi_cell(
            w, 6, "Languages: " + (", ".join(langs) if langs else "N/A")
        )
        pdf.multi_cell(
            w, 6, "Conditions: " + (", ".join(conditions) if conditions else "N/A")
        )
        pdf.multi_cell(
            w, 6, "Emotions: " + (", ".join(emotions) if emotions else "N/A")
        )
        pdf.multi_cell(
            w, 6, f"Turns flagged as risk: {risk_count}"
        )
        pdf.ln(4)

        pdf.set_font("DejaVu", "", 11)
        pdf.multi_cell(
            w,
            6,
            (
                "Ghi chú: PDF này là báo cáo tóm tắt cho mục đích nghiên cứu. "
                "Toàn bộ nội dung hội thoại chi tiết đã được lưu trong file CSV "
                "kèm theo (wellbeing_conversations.csv)."
            ),
        )

        out_path = (reports_dir / f"report_{user_id}.pdf").resolve()
        pdf.output(str(out_path))
        print(f"[OK] PDF exported to: {out_path}")


# -------------------------
# MAIN
# -------------------------

def main():
    print(f"[INFO] DB_PATH = {DB_PATH}")
    print(f"[INFO] FONT_PATH = {FONT_PATH}")

    if not DB_PATH.exists():
        print("[ERR] Database not found")
        return

    messages = load_messages()
    print(f"[INFO] Loaded {len(messages)} messages")

    if not messages:
        print("[WARN] No data to export")
        return

    export_csv(messages)

    # Đảm bảo nếu PDF lỗi thì CSV vẫn ok
    try:
        export_pdf(messages)
    except (fpdf_errors.FPDFException, Exception) as e:
        print("[WARN] PDF export failed, nhưng CSV đã xuất xong.")
        print("       Lý do:", repr(e))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        print("[FATAL] Exception occurred:")
        traceback.print_exc()
