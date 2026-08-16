"""
AI_internals Progress Tracker & Statistics Script
Подсчитывает объем кода (Python LOC), статус завершенности недель и выводит общую статистику.
"""

import sys
from pathlib import Path

# Force UTF-8 stdout encoding on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def analyze_learning_progress():
    base_dir = Path(__file__).parent.parent
    weeks = [
        ("Week 1: LLM Mechanics", base_dir / "week_1_llm_mechanics"),
        ("Week 2: Enterprise RAG", base_dir / "week_2_enterprise_rag"),
        ("Week 3: Agents & Schemas", base_dir / "week_3_agents_structured_outputs"),
        ("Week 4: Evals & MLOps", base_dir / "week_4_evals_finetuning_mlops"),
    ]

    print("=" * 60)
    print(" 🧠 AI INTERNALS — STATISTICAL DASHBOARD")
    print("=" * 60)

    total_py_files = 0
    total_loc = 0

    for name, week_dir in weeks:
        if not week_dir.exists():
            continue

        py_files = list(week_dir.rglob("*.py"))
        loc_count = 0
        for pf in py_files:
            try:
                lines = pf.read_text(encoding="utf-8").splitlines()
                loc_count += len([l for l in lines if l.strip() and not l.strip().startswith("#")])
            except Exception:
                pass

        total_py_files += len(py_files)
        total_loc += loc_count

        print(f"\n📂 {name}")
        print(f"   - Python файлов: {len(py_files)}")
        print(f"   - Чистых строк кода (LOC): {loc_count}")

    print("\n" + "=" * 60)
    print(f"📊 ВСЕГО В КУРСЕ:")
    print(f"   - Всего файлов Python: {total_py_files}")
    print(f"   - Всего строк кода (LOC): {total_loc}")
    print("=" * 60)

if __name__ == "__main__":
    analyze_learning_progress()
