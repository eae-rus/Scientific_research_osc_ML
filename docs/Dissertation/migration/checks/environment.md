# Проверка рабочего окружения и доступных инструментов (A02)

Дата проверки: 2026-09-17  
Сессия: ebd28231-cb5f-4cdb-83f4-0474b4675ce2  
Исполнитель: Antigravity (Gemini 3.8 Flash)

## 1. Доступные рантаймы и интерпретаторы

| Инструмент | Полный путь | Версия | Статус проверки |
|---|---|---|---|
| Python 64-bit | `C:\ProgramData\Anaconda3\python.exe` | 3.13.5 | Проверен запуском команд и скриптов |
| PowerShell 7 | `C:\Program Files\PowerShell\7\pwsh.exe` | 7.4.19 | Проверен в сессии |
| Windows PowerShell | `C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe` | 5.1.19041 | Доступен |
| Git CLI | `C:\Users\Евдаков Алексей\AppData\Local\GitHubDesktop\app-3.2.6\resources\app\git\cmd\git.exe` | 2.43.0.windows.1 | Проверен (`git status`, `git diff`) |

## 2. Средства обработки и рендеринга документов

| Инструмент | Полный путь / пакет | Версия | Назначение и результат проверки |
|---|---|---|---|
| Microsoft Word | `C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE` | 16.0 (build 16.0.14332) | Основной рендерер DOCX; проверен через COM automation |
| Python win32com | В составе Anaconda3 (`win32com.client`) | 308 | Проверено управление Word.Application в фоновом режиме |
| Python docx | `python-docx` | 1.2.0 | Извлечение структуры, абзацев, таблиц и свойств DOCX |
| Pillow (PIL) | `Pillow` | 11.1.0 | Обработка извлечённых изображений и графики |
| Matplotlib | `matplotlib` | 3.10.0 | Генерация и сверка графических материалов |
| pylatexenc | `pylatexenc` | 2.11 | Парсинг LaTeX-макросов и конвертация формул/символов |
| bibtexparser | `bibtexparser` | 2.0.1 | Разбор и генерация файлов библиографии BibTeX |
| Python wrappers | `pdflatex`, `latexcompiler` | 0.1.3 / 1.0 | Обёртки вызова системных TeX-движков (требуют `pdflatex.exe`) |

## 3. Проверка кириллицы и рендеринга неизменённого исходника

1. **Тест кириллицы на изолированном документе**:
   - Создан временный документ с кириллическим текстом в UTF-8.
   - Через Word COM выполнен экспорт в PDF (`ExportAsFixedFormat`).
   - Кириллица отображается корректно, артефактов кодировки нет.

2. **Рендеринг неизменённого baseline DOCX**:
   - Файл: `docs/Dissertation/migration/baseline/Диссертация (Евдаков А.Е.) v0.5.docx`
   - Исходный SHA-256: `50516d7832e6ea133d77b11b48802c4ed1929283ea7601134b878fab0e0338f7`
   - Открытие: `word.Documents.Open(..., ReadOnly=True)`
   - Экспорт: `doc.ExportAsFixedFormat('docs/Dissertation/build/source-render/source-render.pdf', 17)`
   - Время рендера: 11.66 с.
   - Полученный PDF: 8 920 833 байт, ровно 127 страниц.
   - Сверка с архивным PDF (`docs/Dissertation/Архивное/Диссертация (Евдаков А.Е.) v0.5.pdf`): число страниц совпадает (127 страниц).
   - Хеш исходного baseline DOCX после экспорта перепроверен: остался строго неизменным (`50516d78...`).

## 4. Средства компиляции LaTeX (MiKTeX)

Установлен и проверен дистрибутив MiKTeX:
- Путь установки: `C:\Users\Евдаков Алексей\AppData\Local\Programs\MiKTeX\miktex\bin\x64\`
- Версия: MiKTeX 25.12 (установлен 17.09.2026)

| Инструмент | Исполняемый файл | Версия | Статус проверки |
|---|---|---|---|
| `pdflatex` | `pdflatex.exe` | MiKTeX-pdfTeX 4.23 | Проверена компиляция кириллического документа (returncode: 0, PDF 64 КБ) |
| `xelatex` | `xelatex.exe` | MiKTeX-XeTeX 4.16 | Проверена компиляция кириллического документа (returncode: 0, PDF 9.7 КБ) |
| `lualatex` | `lualatex.exe` | MiKTeX-LuaTeX 2.8 | Доступен |
| `biber` | `biber.exe` | 2.21 | Проверен вызовом `--version` |
| `bibtex` | `bibtex.exe` | 0.99d | Доступен |

**Настройка автоустановки пакетов:**
Установлен параметр `[MPM]AutoInstall=1` через `initexmf --set-config-value=[MPM]AutoInstall=1`. Недостающие пакеты и шрифты (включая русские шрифты `lh` и `babel-russian`) загружаются и компилируются автоматически в фоновом режиме без блокирующих запросов.

## 5. Выводы и статус

- Маршруты чтения исходника (Python, python-docx, OOXML ElementTree) и визуального рендеринга исходника (Word COM -> PDF) полностью функционируют и проверены (127 страниц, `build/source-render/source-render.pdf`).
- Локальный компилятор LaTeX (MiKTeX: `pdflatex`, `xelatex`, `biber`) установлен автором, автоматически сконфигурирован и протестирован на пробной компиляции документов с поддержкой русского языка и формул.
- Все критерии карточки A02 («есть воспроизводимые маршруты чтения/рендера и пробной сборки») полностью выполнены.
- Статус A02: **DONE**. Окружение готово как для задач парсинга (A03–A07), так и для пробного переноса (A08) и развёртывания LaTeX-каркаса (B01).
