"""Reproducible conceptual diagrams of the baseline AI-01 architectures.

No trained weights, measured curves or experimental layer sizes are depicted.
The shared drawing primitives are used for all seven figures.
"""
from pathlib import Path
from math import atan2, cos, sin, pi
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript/figures"
FONT = "ArchitectureArial"
W = 680


class Diagram:
    def __init__(self, filename, height):
        self.c = canvas.Canvas(str(OUT / filename), pagesize=(W, height))
        self.c.setTitle(filename.removesuffix(".pdf"))
        self.c.setAuthor("Dissertation architecture diagrams")
        self.h = height

    def box(self, x, y, width, text, height=48, physical=False):
        c = self.c
        c.setStrokeColor(HexColor("#263746"))
        c.setFillColor(HexColor("#E5EFEA" if physical else "#EDF1F5"))
        c.roundRect(x, y, width, height, 5, fill=1, stroke=1)
        lines = text.split("\n")
        c.setFillColor(HexColor("#182B3A"))
        c.setFont(FONT, 15)
        for i, line in enumerate(lines):
            assert pdfmetrics.stringWidth(line, FONT, 15) < width - 10, (line, width)
            c.drawCentredString(x + width / 2, y + height / 2 + (len(lines)-1)*8.5 - i*17 - 5, line)

    def arrow(self, points):
        c = self.c
        c.setStrokeColor(HexColor("#263746"))
        c.setFillColor(HexColor("#263746"))
        c.setLineWidth(1.25)
        path = c.beginPath()
        path.moveTo(*points[0])
        for point in points[1:]:
            path.lineTo(*point)
        c.drawPath(path)
        x, y = points[-1]
        a = atan2(y-points[-2][1], x-points[-2][0])
        p = c.beginPath(); p.moveTo(x, y)
        for sign in [-1, 1]:
            p.lineTo(x-7*cos(a+sign*pi/7), y-7*sin(a+sign*pi/7))
        p.close(); c.drawPath(p, stroke=0, fill=1)

    def text(self, x, y, value, size=14):
        self.c.setFont(FONT, size)
        self.c.setFillColor(HexColor("#263746"))
        self.c.drawCentredString(x, y, value)

    def finish(self):
        self.c.showPage(); self.c.save()


def chain(filename, labels):
    d = Diagram(filename, 105)
    width = 116
    for i, label in enumerate(labels):
        x = 7 + i*137
        d.box(x, 35, width, label)
        if i:
            d.arrow([(x-17, 59), (x, 59)])
    d.finish()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(TTFont(FONT, "C:/Windows/Fonts/arial.ttf"))
    chain("physkan_mlp.pdf", ["Окно C × T", "Вектор C·T", "Linear / BN\nReLU; Dropout", "Следующие\nскрытые слои", "Linear\nK оценок"])
    chain("physkan_cnn.pdf", ["Окно C × T", "Conv1d / BN\nReLU", "MaxPool\nDropout", "Повтор блоков\nУсреднение", "Выход\nK оценок"])

    d = Diagram("physkan_residual.pdf", 180)
    d.box(15, 38, 85, "Вход x")
    d.box(145, 38, 270, "Conv / BN / ReLU / Conv / BN\nОсновной путь F(x)")
    d.box(460, 38, 65, "+")
    d.box(560, 38, 100, "ReLU\nВыход y")
    d.box(180, 118, 225, "Тождественный путь\nили проекция P(x)")
    for points in [[(100,62),(145,62)],[(415,62),(460,62)],[(525,62),(560,62)],[(110,62),(110,142),(180,142)],[(405,142),(493,142),(493,86)]]:
        d.arrow(points)
    d.finish()

    d = Diagram("physkan_kan_edge.pdf", 275)
    d.box(20, 178, 100, "Вход x1")
    d.box(20, 58, 100, "Вход x2")
    for y, label in [(207, "f11(x1)"), (147, "f12(x2)"), (87, "f21(x1)"), (27, "f22(x2)")]:
        d.box(220, y, 165, label, height=38)
    d.box(505, 160, 140, "Сумма\ny1 = f11 + f12")
    d.box(505, 20, 140, "Сумма\ny2 = f21 + f22")
    for points in [[(120,202),(220,226)],[(120,82),(140,82),(140,166),(220,166)],[(120,202),(180,202),(180,106),(220,106)],[(120,82),(150,82),(150,46),(220,46)],[(385,226),(450,226),(450,190),(505,190)],[(385,166),(450,166),(450,178),(505,178)],[(385,106),(465,106),(465,50),(505,50)],[(385,46),(505,38)]]:
        d.arrow(points)
    d.text(340, 257, "Условный KAN-слой; обучаемые функции на каждой связи")
    d.finish()

    d = Diagram("physkan_arithmetic_stem.pdf", 230)
    d.box(12, 94, 116, "Каналы I, U\nC каналов")
    d.box(188, 169, 120, "Произведение\nC/2 каналов", physical=True)
    d.box(188, 94, 120, "Отношение\nC/2 каналов", physical=True)
    d.box(188, 19, 120, "Исходные\nC каналов")
    d.box(355, 169, 115, "BatchNorm")
    d.box(355, 94, 115, "BatchNorm")
    d.box(520, 94, 145, "Объединение\n2C каналов")
    for yy in [193,118,43]:
        d.arrow([(128,118),(155,118),(155,yy),(188,yy)])
    for yy in [193,118]:
        d.arrow([(308,yy),(355,yy)])
        d.arrow([(470,yy),(490,yy),(490,118),(520,118)])
    d.arrow([(308,43),(493,43),(493,103),(520,103)])
    d.text(589, 59, "Далее ConvKAN")
    d.finish()

    d = Diagram("physkan_complex_ops.pdf", 225)
    d.box(15, 87, 135, "Пары A, φ\nДля I и U")
    d.box(210, 145, 260, "Умножение\nA1·A2; φ1 + φ2", physical=True)
    d.box(210, 39, 260, "Деление\nA1 / max(A2, ε); φ1 − φ2", physical=True)
    d.box(535, 87, 130, "Объединение\nс исходными")
    for points in [[(150,111),(175,111),(175,169),(210,169)],[(150,111),(175,111),(175,63),(210,63)],[(470,169),(505,169),(505,125),(535,125)],[(470,63),(505,63),(505,97),(535,97)],[(150,111),(175,111),(175,13),(599,13),(599,87)]]:
        d.arrow(points)
    d.text(340, 212, "При нулевом дополнительном фазовом сдвиге")
    d.finish()

    d = Diagram("physkan_relay_gate.pdf", 305)
    d.box(12, 119, 116, "Признаки\nпары A, φ")
    d.box(175, 211, 147, "Амплитуды\nBatchNorm")
    d.box(175, 85, 147, "Фазы\nBatchNorm")
    d.box(355, 211, 143, "KAN-ветвь A\nsoftplus")
    d.box(355, 123, 143, "KAN-маска\nsigmoid", physical=True)
    d.box(355, 35, 143, "KAN-ветвь φ")
    d.box(540, 211, 110, "Умножение\nA × маска", physical=True)
    d.box(540, 73, 110, "Объединение")
    for points in [[(128,143),(150,143),(150,235),(175,235)],[(128,143),(150,143),(150,109),(175,109)],[(322,235),(355,235)],[(322,109),(337,109),(337,147),(355,147)],[(322,109),(337,109),(337,59),(355,59)],[(498,235),(540,235)],[(498,147),(570,147),(570,211)],[(620,211),(620,121)],[(498,59),(515,59),(515,97),(540,97)]]:
        d.arrow(points)
    d.text(595, 42, "Далее классификатор")
    d.finish()


if __name__ == "__main__":
    main()
