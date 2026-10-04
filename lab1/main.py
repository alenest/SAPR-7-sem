"""
Лабораторная работа №1. Аффинные преобразования.
Студент: Нестеров А.С., группа Аэ-21-23.

Задание: невыпуклый 9-угольник задан кодом Фримэна.
Выполнить перенос по оси Y и отражение относительно прямой y = 3x + 10,
используя отражение относительно оси X. Показать преобразования
пошагово и комплексно.

=== ТЕОРИЯ (сжато) ===
Код Фримэна. Примитив описывается цепочкой элементарных векторов по 8
направлениям. Направление i = i*45°, i = 0..7. Длина вектора
D = T*(sqrt(2))^p, p = i mod 2. При T=1 чётные направления дают
(±1,0)/(0,±1), нечётные — (±1,±1).

Однородные координаты. Точка P(X, Y, W), W != 0. Декартовы:
x = X/W, y = Y/W. При W=1: P(x, y, 1). Зачем: в декартовых 2x2 нельзя
одной матрицей описать перенос, а в однородных 3x3 — можно.

Аффинное преобразование в однородных координатах: p' = M @ p,
где M — матрица 3x3, p = [x, y, 1]^T.

Композиция. Сначала M1, потом M2: p' = M2 @ M1 @ p. Порядок важен.

Смена системы координат. Чтобы выполнить операцию относительно
произвольной точки/прямой, переходим в удобную систему (точка — начало,
прямая — ось X), делаем простое преобразование, возвращаемся обратно.
Именно так строятся rotation_about_point, scaling_about_point и
reflection_about_line.
"""

import math
import os
import numpy as np                      # только для @ и массивов точек
import matplotlib.pyplot as plt         # только для отрисовки


# =============================================================
# ЧАСТЬ 1. КОД ФРИМЭНА
# =============================================================
# 8 направлений обхода:
#     3   2   1
#      \  |  /
#   4 ——  *  —— 0
#      /  |  \
#     5   6   7
FREEMAN_STEPS = {
    '0': ( 1,  0),   # вправо
    '1': ( 1,  1),   # вправо-вверх
    '2': ( 0,  1),   # вверх
    '3': (-1,  1),   # влево-вверх
    '4': (-1,  0),   # влево
    '5': (-1, -1),   # влево-вниз
    '6': ( 0, -1),   # вниз
    '7': ( 1, -1),   # вправо-вниз
}


def decode_freeman(code):
    """Декодирует код Фримэна в список вершин (x, y) без дублирующей точки."""
    x, y = 0, 0
    points = [(x, y)]
    for ch in code:
        dx, dy = FREEMAN_STEPS[ch]
        x += dx
        y += dy
        points.append((x, y))
    return points[:-1]                         # убираем дубликат старта


# =============================================================
# ЧАСТЬ 2. РАБОТА С ТОЧКАМИ И МАТРИЦАМИ
# =============================================================

def points_to_homogeneous(points):
    """Превращает список (x, y) в Nx3 массив [[x, y, 1], ...]."""
    return np.array([[x, y, 1.0] for (x, y) in points])


def apply_matrix(M, points):
    """Применяет матрицу 3x3 к списку точек. Возвращает список (x, y)."""
    P = points_to_homogeneous(points)
    P_new = P @ M.T
    W = P_new[:, 2:3].copy()
    W[W == 0] = 1.0                            # защита от деления на 0
    xy = P_new[:, :2] / W
    return [tuple(p) for p in xy]


# =============================================================
# ЧАСТЬ 3. ЭЛЕМЕНТАРНЫЕ АФФИННЫЕ ПРЕОБРАЗОВАНИЯ (2D)
# =============================================================
# Все матрицы выписаны вручную. Формат — вектор-столбец p = [x, y, 1]^T.

def translation(tx, ty):
    """Перенос на (tx, ty).
    | 1  0  tx |     Последовательные переносы аддитивны:
    | 0  1  ty |     T(a) @ T(b) = T(a+b).
    | 0  0   1 |"""
    return np.array([[1.0, 0.0, float(tx)],
                     [0.0, 1.0, float(ty)],
                     [0.0, 0.0, 1.0]])


def scaling(sx, sy):
    """Масштабирование (sx, sy). При sx=sy — однородное.
    | sx  0  0 |     Последовательные масштабирования мультипликативны.
    |  0 sy  0 |
    |  0  0  1 |"""
    return np.array([[float(sx), 0.0,       0.0],
                     [0.0,      float(sy), 0.0],
                     [0.0,      0.0,       1.0]])


def rotation(theta):
    """Поворот на theta (радианы, против часовой) вокруг начала координат.
    | cos -sin  0 |   Отрицательный поворот — транспонированная матрица
    | sin  cos  0 |   положительного. Два поворота аддитивны.
    |  0    0   1 |"""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def reflection_x():
    """Отражение относительно оси X (y → -y)."""
    return np.array([[1.0,  0.0, 0.0],
                     [0.0, -1.0, 0.0],
                     [0.0,  0.0, 1.0]])


def reflection_y():
    """Отражение относительно оси Y (x → -x)."""
    return np.array([[-1.0, 0.0, 0.0],
                     [ 0.0, 1.0, 0.0],
                     [ 0.0, 0.0, 1.0]])


def reflection_origin():
    """Отражение относительно начала координат."""
    return np.array([[-1.0,  0.0, 0.0],
                     [ 0.0, -1.0, 0.0],
                     [ 0.0,  0.0, 1.0]])


def reflection_y_eq_x():
    """Отражение относительно прямой y = x."""
    return np.array([[0.0, 1.0, 0.0],
                     [1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0]])


def reflection_y_eq_minus_x():
    """Отражение относительно прямой y = -x."""
    return np.array([[ 0.0, -1.0, 0.0],
                     [-1.0,  0.0, 0.0],
                     [ 0.0,  0.0, 1.0]])


# =============================================================
# ЧАСТЬ 4. КОМПОЗИТНЫЕ ПРЕОБРАЗОВАНИЯ (СМЕНА СИСТЕМЫ КООРДИНАТ)
# =============================================================
# Идея: M = T(в удобную систему) @ (простое преобразование) @ T(обратно).

def rotation_about_point(px, py, theta):
    """Поворот на theta вокруг точки (px, py):
    M = T(px, py) @ R(theta) @ T(-px, -py)."""
    return translation(px, py) @ rotation(theta) @ translation(-px, -py)


def scaling_about_point(px, py, sx, sy):
    """Масштабирование относительно точки (px, py):
    M = T(px, py) @ S(sx, sy) @ T(-px, -py)."""
    return translation(px, py) @ scaling(sx, sy) @ translation(-px, -py)


def reflection_about_line(k, b):
    """Отражение относительно прямой y = k*x + b — 5 шагов из лекции:
      1) T(0, -b)    — сдвиг прямой к началу координат по Y
      2) R(-alpha)   — поворот, чтобы прямая совпала с осью X
      3) Rx          — отражение относительно оси X
      4) R(alpha)    — обратный поворот
      5) T(0, b)     — обратный сдвиг
    где alpha = atan(k)."""
    alpha = math.atan(k)
    return (translation(0, b) @ rotation(alpha) @ reflection_x()
            @ rotation(-alpha) @ translation(0, -b))


# =============================================================
# ЧАСТЬ 5. ВИЗУАЛИЗАЦИЯ
# =============================================================

def compute_bounds(polygons, pad=1.5):
    """Габариты по всем полигонам: (x_lo, x_hi, y_lo, y_hi).
    Используется, чтобы задать xlim/ylim и обрезать прямую по краям."""
    xs = [p[0] for poly in polygons for p in poly]
    ys = [p[1] for poly in polygons for p in poly]
    return (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)


def plot_polygon(ax, points, label, color, linestyle='-'):
    """Рисует замкнутый многоугольник и добавляет его в легенду."""
    xs = [p[0] for p in points] + [points[0][0]]
    ys = [p[1] for p in points] + [points[0][1]]
    ax.plot(xs, ys, marker='o', color=color, label=label,
            linestyle=linestyle, linewidth=1.5, markersize=4)


def draw_freeman(ax, points, code):
    """Рисует исходную фигуру в стиле образца: жирный оранжевый контур,
    стрелки на середине каждого ребра, цифры кода и координаты вершин."""
    n = len(points)
    xs = [p[0] for p in points] + [points[0][0]]
    ys = [p[1] for p in points] + [points[0][1]]
    ax.plot(xs, ys, color='orange', linewidth=2.5, zorder=2)
    ax.plot([p[0] for p in points], [p[1] for p in points],
            'o', color='darkorange', markersize=6, zorder=3,
            label='Исходный 9-угольник')

    # Стрелки на середине каждого ребра — по направлению обхода
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        if L < 1e-9:
            continue
        ux, uy = dx / L, dy / L
        ax.annotate('',
                    xy=(mx + ux * 0.4, my + uy * 0.4),
                    xytext=(mx - ux * 0.4, my - uy * 0.4),
                    arrowprops=dict(arrowstyle='-|>', color='crimson',
                                    lw=1.6, mutation_scale=16),
                    zorder=4)

    # Цифры кода рядом с серединой каждого ребра
    for i, ch in enumerate(code):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.18, ch, fontsize=10, color='darkred',
                ha='center', va='bottom', zorder=5,
                bbox=dict(boxstyle='circle,pad=0.18',
                          facecolor='white', edgecolor='darkred', lw=0.8))

    # Координаты вершин
    for (x, y) in points:
        ax.annotate(f'({int(x)},{int(y)})', (x, y),
                    textcoords='offset points', xytext=(6, -10),
                    fontsize=8, color='navy')

    # Габариты исходной фигуры + отступ
    x_lo, x_hi, y_lo, y_hi = compute_bounds([points], pad=1.0)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal')
    ax.set_title(f"Исходная фигура (код Фримэна: {code})")
    ax.legend(loc='upper right')


def draw_line_y_eq_kx_b(ax, k, b, label=None):
    """Рисует прямую y = kx + b по текущим границам осей.
    Т.к. xlim/ylim уже заданы, отрезок прямой автоматически обрежется
    matplotlib'ом по краям области — не нужно считать диапазон самим."""
    x_lo, x_hi = ax.get_xlim()
    ax.plot([x_lo, x_hi],
            [k * x_lo + b, k * x_hi + b],
            '--', color='magenta', linewidth=1.5,
            label=label or f'y = {k}x + {b}',
            zorder=1)


def matrix_to_text(name, M):
    """Красивая текстовая запись матрицы 3x3 для вывода на графике."""
    lines = [name + " ="]
    for row in M:
        lines.append("[ " + "  ".join(f"{v:8.4f}" for v in row) + " ]")
    return "\n".join(lines)


# =============================================================
# ЧАСТЬ 6. ОБРАБОТКА ОДНОЙ ФИГУРЫ
# =============================================================

def process_figure(fig_name, code, output_dir, ty=5, k=3, b=10):
    """Строит 3 картинки для одной фигуры и сохраняет их в output_dir."""
    # 1) Декодируем исходные вершины
    points = decode_freeman(code)

    # 2) Строим нужные матрицы
    T_y    = translation(0, ty)                 # перенос по Y
    M_refl = reflection_about_line(k, b)        # отражение отн. y = kx + b
    M_comb = M_refl @ T_y                       # комплексная матрица

    # Отдельные шаги отражения — для показа пошагово
    T_negb = translation(0, -b)
    R_neg  = rotation(-math.atan(k))
    Rx     = reflection_x()
    R_pos  = rotation(math.atan(k))
    T_posb = translation(0, b)

    # 3) Пошаговые результаты
    p1 = apply_matrix(T_y,    points)           # после переноса
    p2 = apply_matrix(T_negb, p1)               # шаг 1
    p3 = apply_matrix(R_neg,  p2)               # шаг 2
    p4 = apply_matrix(Rx,     p3)               # шаг 3
    p5 = apply_matrix(R_pos,  p4)               # шаг 4
    p6 = apply_matrix(T_posb, p5)               # шаг 5 — итог отражения
    p_comb = apply_matrix(M_comb, points)       # комплексный итог

    # -------- КАРТИНКА 1: исходная фигура с кодом Фримэна --------
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    draw_freeman(ax1, points, code)
    fig1.savefig(os.path.join(output_dir, f"{fig_name}_1_original.png"),
                 dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # -------- КАРТИНКА 2: все шаги вместе + прямая --------
    # ВАЖНО: это ПОШАГОВЫЙ рисунок. Комплексного итога здесь НЕТ.
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    plot_polygon(ax2, points, "Исходный",                   'black')
    plot_polygon(ax2, p1,     f"После переноса T(0,{ty})",  'green')
    plot_polygon(ax2, p2,     f"Шаг 1: T(0,{-b})",          'olive')
    plot_polygon(ax2, p3,     "Шаг 2: R(-alpha)",           'orange')
    plot_polygon(ax2, p4,     "Шаг 3: Rx",                  'red')
    plot_polygon(ax2, p5,     "Шаг 4: R(alpha)",            'purple')
    plot_polygon(ax2, p6,     f"Шаг 5: T(0,{b})",           'brown')

    # Границы по всем полигонам + запас, чтобы прямая была видна рядом
    x_lo, x_hi, y_lo, y_hi = compute_bounds(
        [points, p1, p2, p3, p4, p5, p6], pad=1.5
    )
    ax2.set_xlim(x_lo, x_hi)
    ax2.set_ylim(y_lo, y_hi)

    # Прямая — рисуется поверх заданных границ и обрезается автоматически
    draw_line_y_eq_kx_b(ax2, k, b)

    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_aspect('equal')
    ax2.set_title(f"Пошаговое преобразование: {fig_name}\n"
                  f"(перенос по Y на {ty}, отражение отн. y = {k}x + {b})")
    # Легенда — снаружи осей, чтобы не закрывать полигоны
    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9)
    fig2.savefig(os.path.join(output_dir, f"{fig_name}_2_steps.png"),
                 dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # -------- КАРТИНКА 3: комплексное преобразование + прямая --------
    fig3, ax3 = plt.subplots(figsize=(9, 9))
    plot_polygon(ax3, points, "Исходный",                   'black')
    plot_polygon(ax3, p_comb, "Комплексное преобразование",  'blue')

    x_lo, x_hi, y_lo, y_hi = compute_bounds(
        [points, p_comb], pad=1.5
    )
    ax3.set_xlim(x_lo, x_hi)
    ax3.set_ylim(y_lo, y_hi)

    draw_line_y_eq_kx_b(ax3, k, b)

    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.set_aspect('equal')
    ax3.set_title(f"Комплексное преобразование: {fig_name}\n"
                  f"M_combined = M_reflect @ T(0,{ty})")
    # Легенда — снаружи осей
    ax3.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9)
    # Матрица — в правом нижнем углу осей, чтобы не налезала на фигуры
    ax3.text(0.98, 0.02, matrix_to_text("M_combined", M_comb),
             transform=ax3.transAxes, va='bottom', ha='right',
             family='monospace', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
    fig3.savefig(os.path.join(output_dir, f"{fig_name}_3_combined.png"),
                 dpi=150, bbox_inches='tight')
    plt.close(fig3)


# =============================================================
# ЧАСТЬ 7. ТОЧКА ВХОДА
# =============================================================

if __name__ == "__main__":
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Три невыпуклых 9-угольника: разные формы и масштабы.
    figures = [
        ("figure1", "0072241446366"),              
        ("figure2", "717131454447"),        
        ("figure3", "013000122544446666"),      
    ]

    for fig_name, code in figures:
        print(f"Обработка {fig_name} (код: {code}) ...")
        process_figure(fig_name, code, output_dir,
                       ty=5,        # перенос по Y
                       k=3, b=10)   # отражение отн. y = 3x + 10

    print(f"Готово! Изображения сохранены в папке '{output_dir}'.")