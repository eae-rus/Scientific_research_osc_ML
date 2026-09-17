# Инструменты и проверенные команды запуска

Все команды запускаются из корня репозитория (`Scientific_research_osc_ML`) в среде PowerShell 7 (`pwsh`) или Windows PowerShell.  
Кодировка всех исходных текстовых файлов, скриптов и отчётов — **UTF-8 без BOM**.

## 1. Запуск Python и служебных скриптов

Используется среда Anaconda3 с Python 3.13.5 (64-bit):

```powershell
& "C:\ProgramData\Anaconda3\python.exe" <путь_к_скрипту>
```

Пример запуска инвентаризации источников:
```powershell
& "C:\ProgramData\Anaconda3\python.exe" docs/Dissertation/tools/inventory_sources.py
```

## 2. Управление версиями (Git)

Используется установленный Git CLI:

```powershell
& "C:\Users\Евдаков Алексей\AppData\Local\GitHubDesktop\app-3.2.6\resources\app\git\cmd\git.exe" status
```

Для проверки статуса только папки диссертации:
```powershell
& "C:\Users\Евдаков Алексей\AppData\Local\GitHubDesktop\app-3.2.6\resources\app\git\cmd\git.exe" status --porcelain "docs/Dissertation"
```

## 3. Рендеринг и экспорт DOCX в PDF

Экспорт неизменённого DOCX выполняется через Microsoft Word COM automation в режиме `ReadOnly`:

```powershell
& "C:\ProgramData\Anaconda3\python.exe" -c "
import win32com.client
from pathlib import Path
root = Path('.').resolve()
docx_path = (root / 'docs/Dissertation/migration/baseline/Диссертация (Евдаков А.Е.) v0.5.docx').resolve()
pdf_path = (root / 'docs/Dissertation/build/source-render/source-render.pdf').resolve()
pdf_path.parent.mkdir(parents=True, exist_ok=True)
word = win32com.client.DispatchEx('Word.Application')
word.Visible = False
try:
    doc = word.Documents.Open(FileName=str(docx_path), ReadOnly=True)
    doc.ExportAsFixedFormat(str(pdf_path), 17) # 17 = wdExportFormatPDF
    doc.Close(False)
finally:
    word.Quit()
print('Export completed:', pdf_path.exists(), pdf_path.stat().st_size)
"
```

Результат сохраняется в `docs/Dissertation/build/source-render/source-render.pdf`.

## 4. Средства парсинга LaTeX и BibTeX (Python)

В среде Python доступны библиотеки чистого парсинга:
- `pylatexenc` (2.11) — разбор LaTeX-макросов, конвертация формул и спецсимволов.
- `bibtexparser` (2.0.1) — чтение, фильтрация и генерация записей `.bib`.

## 5. Компиляция LaTeX (MiKTeX)

На хосте установлен дистрибутив **MiKTeX 25.12** (каталог `C:\Users\Евдаков Алексей\AppData\Local\Programs\MiKTeX\miktex\bin\x64\`).  
Автоматическая установка недостающих пакетов включена: `[MPM]AutoInstall=1`.

Проверенные команды сборки:

```powershell
# Сборка через pdflatex
pdflatex -interaction=nonstopmode <имя_файла>.tex

# Сборка через xelatex (рекомендуется для UTF-8 и русских шрифтов)
xelatex -interaction=nonstopmode <имя_файла>.tex

# Обработка библиографии
biber <имя_файла>
```
