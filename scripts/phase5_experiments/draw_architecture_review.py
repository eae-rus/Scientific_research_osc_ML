"""Generate explanatory figures for the October 2026 architecture review."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

DEST = Path(__file__).resolve().parents[2] / "docs/phase_discription/phase_5/figures"
DEST.mkdir(exist_ok=True)


def canvas(title, subtitle):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set(xlim=(0, 14), ylim=(0, 8))
    ax.axis("off")
    ax.text(.35, 7.6, title, fontsize=21, weight="bold", color="#17324d")
    ax.text(.35, 7.15, subtitle, fontsize=12, color="#506478")
    return fig, ax


def box(ax, x, y, w, h, text, color="#e7f0fa", size=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", facecolor=color, edgecolor="#66829c", linewidth=1.4))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=size, linespacing=1.5)


def arrow(ax, start, end, label=None):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="#3a536c", lw=1.7))
    if label:
        ax.text((start[0]+end[0])/2, (start[1]+end[1])/2+.12, label, fontsize=10, ha="center")


def save(fig, name):
    fig.savefig(DEST / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


fig, ax = canvas("Текущая модель: Physical KAN-Transformer", "Глобальные SSL-опыты small / heavy: контракт B, пять снимков окна")
box(ax,.4,5.5,3.2,1,"Фазоры B: 156 признаков\nNaN + provenance")
box(ax,4.2,5.5,3.2,1,"DataSanitizer\nПодстановка missing_token")
box(ax,8,5.5,5.2,1,"PhysicalStem\nОбщие KAN и complex selector")
arrow(ax,(3.65,6),(4.1,6)); arrow(ax,(7.45,6),(7.9,6))
box(ax,8,3.7,5.2,1.1,"concat(KAN, interaction) → projection\nДве скрытые половины", "#fff0da")
arrow(ax,(10.6,5.4),(10.6,4.9))
box(ax,4.2,3.7,3.2,1.1,"+ provenance context\n+ position encoding")
arrow(ax,(7.9,4.25),(7.5,4.25))
box(ax,.4,3.7,3.2,1.1,"L × encoder\nAttention + physical FFN")
arrow(ax,(4.1,4.25),(3.7,4.25))
box(ax,.4,1.8,3.2,1,"Final norm\nSSL reconstruction head")
arrow(ax,(2,3.6),(2,2.9))
box(ax,4.2,1.6,9,1.35,"Проверенные ограничения\nНет отдельного raw_A bypass; selector смешивает физические каналы.\nНормализованные скрытые координаты поступают в torch.polar.", "#fde9e5")
ax.text(.4,.55,"Residual внутри encoder исправен. Физический смысл скрытых половин не гарантирован.",fontsize=13,color="#933d34")
save(fig,"architecture_current_20261028.png")

fig, ax = canvas("Residual: что проходит по прямому пути", "Pre-LN блок сохраняет скрытый x; это не отдельный обход исходных измерений")
box(ax,.5,4.6,1.5,1,"x")
box(ax,3,4.6,3,1,"Norm → Attention")
box(ax,7,4.6,1.5,1,"+ → u")
box(ax,9.4,4.6,3.8,1,"u = x + Attention(Norm(x))",size=10)
arrow(ax,(2.1,5.1),(2.9,5.1)); arrow(ax,(6.1,5.1),(6.9,5.1))
ax.plot([1.25,1.25,7.75],[5.7,6.3,6.3],color="#3a536c",lw=1.7)
arrow(ax,(7.75,6.3),(7.75,5.7)); arrow(ax,(8.6,5.1),(9.3,5.1))
box(ax,.5,2.5,1.5,1,"u")
box(ax,3,2.5,3,1,"Norm → Physical FFN")
box(ax,7,2.5,1.5,1,"+ → y")
box(ax,9.4,2.5,3.8,1,"y = u + FFN(Norm(u))",size=11)
arrow(ax,(2.1,3),(2.9,3)); arrow(ax,(6.1,3),(6.9,3)); arrow(ax,(8.6,3),(9.3,3))
ax.plot([1.25,1.25,7.75],[2.4,1.9,1.9],color="#3a536c",lw=1.7)
arrow(ax,(7.75,1.9),(7.75,2.4))
ax.text(.5,.85,"Если обе ветви равны нулю: y = x,  dy/dx = I. Численный тест: обе ошибки 0.",fontsize=14,color="#22634e")
save(fig,"architecture_residual_20261028.png")

fig, ax = canvas("VNext: физические величины и скрытые признаки", "Предлагаемая архитектура, ещё не реализована и не обучена")
box(ax,.4,5.25,3,1.25,"Physical state\nU / I, гармоники, masks\nСогласованные единицы", "#e4f3ea")
box(ax,4.1,5.25,4,1.25,"Поканальные KAN + время\nРанние U и I независимы")
box(ax,8.8,5.25,4.3,1.25,"Локальная подробная ветвь\nСетка около 1/8 периода")
arrow(ax,(3.5,5.9),(4,5.9)); arrow(ax,(8.2,5.9),(8.7,5.9))
box(ax,.4,3.05,3,1.3,"Типизированная физика\nS = U · conj(I)\nZ / Y + validity", "#e4f3ea")
arrow(ax,(1.9,5.15),(1.9,4.45))
box(ax,4.1,3.05,4,1.3,"Отдельные проекторы семейств\nconcat + raw bypass + mixer", "#fff0da")
arrow(ax,(3.5,3.7),(4,3.7)); arrow(ax,(6.1,5.15),(6.1,4.45))
box(ax,8.8,3.05,4.3,1.3,"Summary-токены\nГлубокий глобальный KAN\nКонтекст до 30 секунд")
arrow(ax,(10.95,5.15),(10.95,4.45)); arrow(ax,(8.2,3.7),(8.7,3.7))
box(ax,4.1,.95,9,1.2,"Объединение локальных признаков и глобального контекста\nSSL Re/Im → отдельные головы РНМ / ОЗЗ\nOnline: только причинные связи и summaries", "#e8e8f6")
arrow(ax,(6.1,2.95),(6.1,2.25)); arrow(ax,(10.95,2.95),(10.95,2.25))
ax.text(.4,.35,"Physical state остаётся доступным; latent может быть знаковым и не объявляется физическим фазором.",fontsize=12,color="#506478")
save(fig,"architecture_next_20261028.png")
