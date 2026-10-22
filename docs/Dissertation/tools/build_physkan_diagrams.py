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

    def box(self, x, y, width, text, height=48, physical=False, size=15):
        c = self.c
        c.setStrokeColor(HexColor("#263746"))
        c.setFillColor(HexColor("#E5EFEA" if physical else "#EDF1F5"))
        c.roundRect(x, y, width, height, 5, fill=1, stroke=1)
        lines = text.split("\n")
        c.setFillColor(HexColor("#182B3A"))
        c.setFont(FONT, size)
        for i, line in enumerate(lines):
            assert pdfmetrics.stringWidth(line, FONT, size) < width - 10, (line, width)
            c.drawCentredString(x + width / 2, y + height / 2 + (len(lines)-1)*size*.57 - i*size*1.14 - size/3, line)

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


def transformer_details():
    """Exact base forward paths, with named connectors instead of crossed wires.

    Drawings omit dropout except where noted. They do not depict measured
    weights, historical checkpoint identity, optional ablations or future heads.
    """
    d = Diagram('transformer_baseline_detail.pdf', 310)
    for x,w,t in [(10,140,'Спектральные\nвекторы зон'),(180,260,'Linear → GELU → LayerNorm\nВходной блок'),(480,190,'Добавление\nположения зоны')]:
        d.box(x,245,w,t,size=14)
    d.arrow([(150,269),(180,269)]); d.arrow([(440,269),(480,269)])
    d.text(340,214,'Один повторяемый блок; вход H',14)
    items=[(10,80,'Норма'),(110,120,'Многоголовое\nвнимание'),(250,35,'+'),(305,80,'Норма'),(405,120,'Linear\nGELU → Linear'),(545,35,'+')]
    for x,w,t in items:d.box(x,110,w,t,size=12)
    for a,b in [(90,110),(230,250),(285,305),(385,405),(525,545)]:d.arrow([(a,134),(b,134)])
    d.arrow([(3,134),(10,134)])
    d.arrow([(3,134),(3,183),(267,183),(267,158)])
    d.arrow([(295,134),(295,196),(562,196),(562,158)])
    d.arrow([(580,134),(645,134)])
    d.text(340,80,'Обходные связи передают вход к покоординатному сложению',13)
    d.box(10,12,660,'После последнего блока: LayerNorm → усреднение внутри зоны → классификатор',size=13)
    d.text(340,232,'Dropout опущен; нормализация охватывает весь скрытый вектор',12)
    d.finish()

    d=Diagram('transformer_stem_detail.pdf',420)
    d.text(340,398,'PhysicalStem: общие входы и три назначения угла',17)
    d.box(10,230,115,'Исходные\nпары A, φ',height=60)
    d.box(155,230,130,'Разделение\nφ′ = φ + b',height=60)
    d.arrow([(125,260),(155,260)])
    for y,t in [(320,'A × g(φ′)\nзатем FastKAN'),(220,'A, φ′ → комплексный блок'),(120,'Углы φ′ без KAN')]:
        d.box(325,y,200,t,physical=y!=120,size=14)
        d.arrow([(285,260),(305,260),(305,y+24),(325,y+24)])
    for y,t in [(320,'Отклики h'),(220,'Модули M\nАргументы ψ'),(120,'Сдвинутые углы φ′')]:
        d.box(550,y,125,t,size=12)
        d.arrow([(525,y+24),(550,y+24)])
    d.text(340,100,'Сборка по именованным выходам; квадратные скобки означают размещение рядом',12)
    d.box(10,25,120,'[h, M]',size=15);d.box(150,25,110,'Проекция u',size=13)
    d.box(295,25,120,'[φ′, ψ]',size=15);d.box(435,25,110,'Проекция v',size=13)
    d.box(575,25,100,'[u | v]',size=16)
    d.arrow([(130,49),(150,49)]);d.arrow([(415,49),(435,49)])
    d.arrow([(260,49),(275,49),(275,12),(625,12),(625,25)])
    d.arrow([(545,49),(575,49)])
    d.text(340,379,'Комплексный блок получает A до направляющего умножения',13)
    d.finish()

    d=Diagram('transformer_interactions_detail.pdf',370)
    d.text(340,348,'Комплексные взаимодействия: произведения и отношения параллельны',16)
    d.box(10,150,110,'Комплексные\nвеличины z',height=65,size=14)
    d.box(150,140,150,'Общие вещественные\nвеса W для Re и Im\ny = Wz',height=85,size=12)
    d.arrow([(120,182),(150,182)])
    for y,t in [(280,'Группа 1'),(220,'Группа 2'),(100,'Группа 3'),(40,'Группа 4')]:
        d.box(335,y,95,t,height=38,size=13)
        d.arrow([(300,182),(317,182),(317,y+19),(335,y+19)])
    d.box(480,237,190,'Попарное произведение\np = y1 · y2',height=65,physical=True,size=14)
    d.box(480,42,190,'Попарное отношение\ny3 · conj(y4) / max(|y4|², ε²)\nЗатем tanh модуля',height=93,physical=True,size=12)
    for pts in [[(430,299),(455,299),(455,284),(480,284)],[(430,239),(455,239),(455,254),(480,254)],[(430,119),(480,119)],[(430,59),(480,59)]]:d.arrow(pts)
    d.text(340,16,'Выход [p | d]; при n результатах по n/2 в каждой ветви. Исходные z сюда не добавляются',12)
    d.text(340,324,'Входной каскад использует модули и аргументы; FFN использует модули',13)
    d.finish()

    d=Diagram('transformer_attention_detail.pdf',320)
    d.text(340,298,'Полярное многоголовое внимание; верхняя часть для одной головы',16)
    for y,t,p in [(220,'Часть u','Q_A, K_A'),(110,'Часть v','Q_φ, K_φ')]:
        d.box(10,y,90,t,size=14);d.box(125,y,130,'Запросы и ключи\n'+p,size=13)
        d.box(285,y,120,'Скалярная\nоценка сходства',size=12)
        d.arrow([(100,y+24),(125,y+24)]);d.arrow([(255,y+24),(285,y+24)])
    d.box(450,166,40,'+',size=19)
    d.arrow([(405,244),(430,244),(430,200),(450,200)])
    d.arrow([(405,134),(430,134),(430,180),(450,180)])
    d.box(520,155,150,'Масштаб и маска\nОбщий Softmax\nВеса α',height=70,size=13)
    d.arrow([(490,190),(520,190)])
    for x,w,t in [(10,135,'Значения\nV_A и V_φ'),(175,155,'Одни веса α\nк обеим ветвям'),(360,175,'Сборка голов\nРаздельные проекции'),(565,105,'[Δu | Δv]')]:
        d.box(x,20,w,t,size=12)
    for a,b in [(145,175),(330,360),(535,565)]:d.arrow([(a,44),(b,44)])
    d.arrow([(595,155),(678,155),(678,89),(252,89),(252,68)])
    d.text(340,278,'Значения V также получают обучаемой проекцией u и v; направляющего блока здесь нет',12)
    d.finish()

    d=Diagram('transformer_ffn_detail.pdf',345)
    d.text(340,323,'Физический FFN: два амплитудных вклада и отдельное приращение угла',16)
    d.box(10,155,110,'Вход\n[u | v]',height=60)
    d.box(165,245,215,'KAN1 → KAN2\nВход u',physical=True,size=14)
    d.box(165,100,215,'Комплексный блок (u, v)\nМодули → проекция',physical=True,size=13)
    for y in [269,124]:d.arrow([(120,185),(140,185),(140,y),(165,y)])
    d.box(405,245,55,'×',size=20);d.box(405,100,55,'×',size=20)
    d.arrow([(380,269),(405,269)]);d.arrow([(380,124),(405,124)])
    d.box(385,185,95,'g(v)',height=35,size=14);d.arrow([(432,220),(432,245)])
    d.arrow([(120,185),(150,185),(150,202),(385,202)])
    d.box(340,40,185,'Скаляр sigmoid(γ)',height=35,size=14);d.arrow([(432,75),(432,100)])
    d.box(520,169,40,'+',size=20)
    d.arrow([(460,269),(490,269),(490,202),(520,202)])
    d.arrow([(460,124),(490,124),(490,182),(520,182)])
    d.box(585,169,85,'[Δu | b]',size=14);d.arrow([(560,193),(585,193)])
    d.box(570,250,100,'Обучаемый b',height=35,size=12);d.arrow([(620,250),(620,217)])
    d.text(340,18,'Снаружи FFN: прежние u и v плюс [Δu | b]. Dropout в KAN-ветви опущен',12)
    d.text(340,302,'Коэффициент g(v) и скаляр sigmoid(γ) имеют разное назначение',13)
    d.text(231,173,'Угол v идёт также в g(v)',12)
    d.finish()

    d=Diagram('transformer_gate_geometry.pdf',335)
    d.text(340,313,'Плавное угловое управление модулем',18)
    c=d.c;cx,cy,R=160,167,102
    c.setStrokeColor(HexColor('#B2BAC1'));c.setDash(3,3);c.circle(cx,cy,R);c.setDash()
    d.arrow([(40,cy),(285,cy)]);d.arrow([(cx,47),(cx,282)])
    d.text(284,cy-17,'Re',12);d.text(cx+16,281,'Im',12)
    mu=pi/6
    c.setDash(3,3);c.setStrokeColor(HexColor('#777777'));c.line(cx,cy,cx+R*cos(mu),cy+R*sin(mu));c.setDash()
    d.text(266,217,'Центр μ',12)
    for angle in [pi/2,5*pi/6]:
        coeff=.2+.8/(1+__import__('math').exp(-8*(cos(angle-mu)-cos(pi/3))))
        c.setStrokeColor(HexColor('#999999'));c.setDash(3,3);c.line(cx,cy,cx+R*cos(angle),cy+R*sin(angle));c.setDash()
        # Explicit solid output arrow, same angle, shortened radius.
        c.setStrokeColor(HexColor('#0A6C9D'));c.setFillColor(HexColor('#0A6C9D'))
        xx=cx+R*coeff*cos(angle);yy=cy+R*coeff*sin(angle)
        c.setLineWidth(2);c.line(cx,cy,xx,yy)
        a=angle;p=c.beginPath();p.moveTo(xx,yy);p.lineTo(xx-7*cos(a+.3),yy-7*sin(a+.3));p.lineTo(xx-7*cos(a-.3),yy-7*sin(a-.3));p.close();c.drawPath(p,stroke=0,fill=1)
    d.text(116,54,'Пунктир — вход A = 1',12);d.text(145,35,'Синий — выход A · g(φ)',12)
    x0,y0,ww,hh=365,84,290,160
    d.arrow([(x0,y0),(x0+ww+8,y0)]);d.arrow([(x0,y0),(x0,y0+hh+14)])
    c.setStrokeColor(HexColor('#0A6C9D'));c.setLineWidth(2);p=c.beginPath()
    for i in range(361):
        delta=-pi+2*pi*i/360;g=.2+.8/(1+__import__('math').exp(-8*(cos(delta)-.5)))
        xx=x0+ww*i/360;yy=y0+hh*g
        if i==0:p.moveTo(xx,yy)
        else:p.lineTo(xx,yy)
    c.drawPath(p)
    for angle,t in [(-180,'−180°'),(-60,'−60°'),(0,'0°'),(60,'60°'),(180,'180°')]:d.text(x0+ww*(angle+180)/360,y0-19,t,11)
    d.text(354,y0+hh,'1',11);d.text(346,y0+hh*.2,'0,2',11)
    d.text(510,265,'Коэффициент g(φ)',14);d.text(510,42,'Отклонение δ = φ − μ',13)
    d.text(340,8,'Условный пример: полуширина 60°, нижний уровень 0,2. Не характеристика обученной модели',12)
    d.text(340,290,'Само умножение меняет длину вектора, но не его угол',14)
    d.finish()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(TTFont(FONT, "C:/Windows/Fonts/arial.ttf"))
    import sys
    if '--transformer-details' in sys.argv:
        transformer_details()
        return
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
