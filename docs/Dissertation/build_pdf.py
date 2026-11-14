"""Сборка PDF диссертации. Запуск: python build_pdf.py --help."""

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4


ROOT = Path(__file__).resolve().parent
MANUSCRIPT = ROOT / "manuscript"
OUTPUT = ROOT / "build" / "reading" / "latex"
CHAPTER_COUNT = 5
PENDING = re.compile(
    r"undefined references|undefined citations"
    r"|Label\(s\) may have changed|Rerun to get",
    re.IGNORECASE,
)


def compile_pdf(engine, work, comments):
    job = "dissertation_" + uuid4().hex
    entry = (
        r"\def\AuthorReview{1}\input{main.tex}"
        if comments
        else r"\AtBeginDocument{\renewcommand{\commentnote}[3]{}"
        r"\renewcommand{\wordrevnote}[4]{}}\input{main.tex}"
    )
    for attempt in range(1, 6):
        print(f"{'С примечаниями' if comments else 'Без примечаний'}, проход {attempt}", flush=True)
        result = subprocess.run(
            [engine, "--disable-installer", "-interaction=nonstopmode",
             "-halt-on-error", "-file-line-error", f"-output-directory={work}",
             f"-jobname={job}", entry],
            cwd=MANUSCRIPT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log = work / f"{job}.log"
        if result.returncode:
            tail = result.stdout.decode("utf-8", errors="replace").splitlines()[-30:]
            raise RuntimeError("Ошибка LaTeX:\n" + "\n".join(tail))
        if not PENDING.search(log.read_text(encoding="utf-8", errors="replace")):
            return work / f"{job}.pdf"
    raise RuntimeError("После пяти проходов остались предупреждения о ссылках.")


def extract_chapters(source, destination, chapters):
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(source)
    labels = dict(reader.named_destinations)
    starts = [reader.get_destination_page_number(labels[f"chapter.{n}"])
              for n in range(1, CHAPTER_COUNT + 1)]
    # The bibliography is the first unnumbered chapter after the last numbered one.
    end = min(
        reader.get_destination_page_number(dest)
        for key, dest in labels.items()
        if key.startswith("chapter*.")
        and reader.get_destination_page_number(dest) > starts[-1]
    )
    boundaries = starts + [end]
    pages = [page for n in chapters for page in range(boundaries[n - 1], boundaries[n])]
    writer = PdfWriter()
    writer.append(reader, pages=pages, import_outline=False)

    # Omit links to destinations outside the extracted pages.
    local = writer.named_destinations
    for page in writer.pages:
        annotations = page.get("/Annots", [])
        for index in range(len(annotations) - 1, -1, -1):
            link = annotations[index].get_object()
            action = link.get("/A", {})
            target = action.get("/D") if action.get("/S") == "/GoTo" else link.get("/Dest")
            if isinstance(target, str) and target not in local:
                del annotations[index]
    writer.write(destination)


def main(*, default_chapters=None, default_mode="comments"):
    parser = argparse.ArgumentParser(description="Сборка диссертации в build/reading/latex.")
    parser.add_argument("--chapters", nargs="+", type=int, choices=range(1, CHAPTER_COUNT + 1),
                        default=default_chapters,
                        help="Номера глав через пробел. Переопределяет уставку CHAPTERS.")
    parser.add_argument("--mode", choices=("comments", "plain", "both"), default=default_mode,
                        help="С примечаниями, без примечаний или обе версии.")
    args = parser.parse_args()
    if args.chapters and any(
        type(chapter) is not int or not 1 <= chapter <= CHAPTER_COUNT
        for chapter in args.chapters
    ):
        parser.error(f"CHAPTERS: укажите целые номера от 1 до {CHAPTER_COUNT}.")
    if args.mode not in ("comments", "plain", "both"):
        parser.error('MODE: выберите "comments", "plain" или "both".')
    engine = shutil.which("pdflatex")
    if not engine:
        parser.error("pdflatex не найден в PATH. Добавьте каталог MiKTeX.")
    if args.chapters:
        try:
            import pypdf  # noqa: F401
        except ImportError:
            parser.error("Для отдельных глав установите pypdf: python -m pip install pypdf")

    chapters = sorted(set(args.chapters or []))
    basename = "chapters_" + "_".join(map(str, chapters)) if chapters else "dissertation"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.mode != "plain":
        subprocess.run([sys.executable, str(ROOT / "tools" / "track_author_notes.py"), "--review-tex"],
                       cwd=ROOT, stdout=subprocess.DEVNULL, check=True)

    with TemporaryDirectory(prefix="dissertation-pdf-") as directory:
        work = Path(directory)
        for mode in ("comments", "plain"):
            if args.mode not in (mode, "both"):
                continue
            pdf = compile_pdf(engine, work, comments=mode == "comments")
            name = f"{basename}_{mode}.pdf"
            if chapters:
                extracted = work / name
                extract_chapters(pdf, extracted, chapters)
                pdf = extracted
            destination = OUTPUT / name
            staging = OUTPUT / f".{uuid4().hex}.pdf.tmp"
            try:
                shutil.copyfile(pdf, staging)
                staging.replace(destination)
            finally:
                staging.unlink(missing_ok=True)
            print(destination, flush=True)


if __name__ == "__main__":
    # РУЧНОЙ ЗАПУСК. Измените две уставки и запустите файл через F5 / Run Python File.
    # CHAPTERS: None — вся диссертация, (4,) — глава 4, (4, 5) — главы 4 и 5.
    # MODE: "comments" — с примечаниями, "plain" — без, "both" — обе версии.
    # PDF появятся в build/reading/latex. Явные аргументы CLI имеют приоритет.
    CHAPTERS = None
    MODE = "comments"

    try:
        main(default_chapters=CHAPTERS, default_mode=MODE)
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError, KeyError) as error:
        sys.exit(str(error))
