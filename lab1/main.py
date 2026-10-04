# -*- coding: utf-8 -*-
# Импортируем math для тригонометрии (cos, sin, atan)
import math
# Импортируем matplotlib ТОЛЬКО для отрисовки (не для преобразований)
import matplotlib.pyplot as plt
# Импортируем os для работы с папками и путями
import os


# =============================================================
# ЧАСТЬ 1. КОД ФРИМЭНА
# =============================================================

# Словарь: цифра кода -> единичный вектор (dx, dy)
FREEMAN_STEPS = {
    '0': (1, 0),    # вправо
    '1': (1, 1),    # вправо-вверх
    '2': (0, 1),    # вверх
    '3': (-1, 1),   # влево-вверх
    '4': (-1, 0),   # влево
    '5': (-1, -1),  # влево-вниз
    '6': (0, -1),   # вниз
    '7': (1, -1),   # вправо-вниз
}


def decode_freeman(code):
    """Преобразует строку кода Фримэна в список вершин (x, y)."""
    x, y = 0, 0                     # стартовая точка обхода
    points = [(x, y)]               # список вершин, первая — стартовая
    for ch in code:                 # перебираем каждый символ кода
        dx, dy = FREEMAN_STEPS[ch]  # получаем вектор шага
        x += dx                     # сдвигаемся по X
        y += dy                     # сдвигаемся по Y
        points.append((x, y))       # добавляем новую вершину
    return points                   # возвращаем все вершины


# =============================================================
# ЧАСТЬ 2. МАТРИЦЫ 3x3 ВРУЧНУЮ
# =============================================================

def matmul(A, B):
    """Умножение двух матриц 3x3 (реализовано вручную, без библиотек)."""
    R = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]   # заготовка результата 3x3
    for i in range(3):                       # строки A
        for j in range(3):                   # столбцы B
            s = 0                            # накопитель суммы
            for k in range(3):               # общий индекс
                s += A[i][k] * B[k][j]       # добавляем произведение
            R[i][j] = s                      # записываем элемент
    return R                                 # возвращаем матрицу


def apply_matrix(M, points):
    """Применяет матрицу 3x3 к списку точек (x, y)."""
    result = []                              # новый список точек
    for (x, y) in points:                    # перебираем все точки
        nx = M[0][0] * x + M[0][1] * y + M[0][2]   # новая координата X
        ny = M[1][0] * x + M[1][1] * y + M[1][2]   # новая координата Y
        result.append((nx, ny))              # добавляем в результат
    return result                            # возвращаем список


# =============================================================
# ЧАСТЬ 3. ЭЛЕМЕНТАРНЫЕ МАТРИЦЫ ПРЕОБРАЗОВАНИЙ
# =============================================================

def translation(tx, ty):
    """Матрица переноса на вектор (tx, ty)."""
    return [[1, 0, tx],                      # строка 1
            [0, 1, ty],                      # строка 2
            [0, 0, 1]]                       # строка 3


def rotation(theta):
    """Матрица поворота на угол theta (радианы, против часовой)."""
    c = math.cos(theta)                      # косинус угла
    s = math.sin(theta)                      # синус угла
    return [[c, -s, 0],                      # строка 1
            [s,  c, 0],                      # строка 2
            [0,  0, 1]]                      # строка 3


def reflection_x():
    """Матрица отражения относительно оси X."""
    return [[1,  0, 0],                      # строка 1
            [0, -1, 0],                      # строка 2
            [0,  0, 1]]                      # строка 3


def reflection_line(k, b):
    """Матрица отражения относительно прямой y = k*x + b."""
    alpha = math.atan(k)                                              # угол наклона прямой
    T1 = translation(0, -b)                                           # шаг 1: T(0, -b)
    R1 = rotation(-alpha)                                             # шаг 2: R(-alpha)
    Rx = reflection_x()                                               # шаг 3: отражение отн. OX
    R2 = rotation(alpha)                                              # шаг 4: R(alpha)
    T2 = translation(0, b)                                            # шаг 5: T(0, b)
    M = matmul(T2, matmul(R2, matmul(Rx, matmul(R1, T1))))            # M = T2*R2*Rx*R1*T1
    return M                                                          # возвращаем матрицу


# =============================================================
# ЧАСТЬ 4. ВИЗУАЛИЗАЦИЯ
# =============================================================

def plot_polygon(ax, points, label, color):
    """Рисует замкнутый многоугольник и добавляет его в легенду."""
    xs = [p[0] for p in points]              # список X-координат
    ys = [p[1] for p in points]              # список Y-координат
    xs.append(points[0][0])                  # замыкаем контур по X
    ys.append(points[0][1])                  # замыкаем контур по Y
    ax.plot(xs, ys, marker='o', color=color, label=label)   # рисуем линию и вершины


def draw_freeman(ax, points, code):
    """Красиво рисует исходную фигуру с кодом Фримэна, стрелками и подписями."""
    pts = points + [points[0]]               # замыкаем список вершин
    plot_polygon(ax, points, "Исходный многоугольник", 'black')     # контур
    for i in range(len(code)):               # проходим по всем шагам
        x1, y1 = pts[i]                      # начало i-го отрезка
        x2, y2 = pts[i + 1]                  # конец i-го отрезка
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 # середина отрезка
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),               # рисуем стрелку
                    arrowprops=dict(arrowstyle='->', color='orange', lw=1))
        ax.text(mx, my, code[i], fontsize=11, color='red',          # цифра кода
                ha='center', va='center')
    for (x, y) in points:                    # подписываем координаты вершин
        ax.annotate(f'({x},{y})', (x, y), textcoords='offset points',
                    xytext=(5, 5), fontsize=8, color='blue')
    ax.grid(True, linestyle='--', alpha=0.5) # координатная сетка
    ax.set_aspect('equal')                   # равный масштаб по осям
    ax.set_title("Исходная фигура (код Фримэна: " + code + ")")     # заголовок
    ax.legend(loc='upper right')             # легенда


def matrix_to_text(name, M):
    """Формирует красивую текстовую запись матрицы 3x3."""
    lines = [name + " ="]                    # заголовок
    for row in M:                            # перебираем строки матрицы
        lines.append("[ " + "  ".join(f"{v:7.3f}" for v in row) + " ]")
    return "\n".join(lines)                  # объединяем в один текст


# =============================================================
# ЧАСТЬ 5. ОБРАБОТКА ОДНОЙ ФИГУРЫ
# =============================================================

def process_figure(fig_name, code, output_dir, ty=5, k=3, b=10):
    """Строит 3 картинки для одной фигуры и сохраняет их."""
    # 1) Декодируем код Фримэна (без последней дублирующей точки)
    points = decode_freeman(code)[:-1]                        # 9 вершин

    # 2) Строим матрицы преобразований
    T_y    = translation(0, ty)                               # перенос по Y
    T_negb = translation(0, -b)                               # шаг 1 отражения
    R_neg  = rotation(-math.atan(k))                          # шаг 2 отражения
    Rx     = reflection_x()                                   # шаг 3 отражения
    R_pos  = rotation(math.atan(k))                           # шаг 4 отражения
    T_posb = translation(0, b)                                # шаг 5 отражения
    M_refl = matmul(T_posb, matmul(R_pos, matmul(Rx, matmul(R_neg, T_negb))))
    M_comb = matmul(M_refl, T_y)                              # комплексная матрица

    # 3) Пошаговые результаты
    p1 = apply_matrix(T_y,    points)                         # после переноса
    p2 = apply_matrix(T_negb, p1)                             # после T(0,-b)
    p3 = apply_matrix(R_neg,  p2)                             # после R(-alpha)
    p4 = apply_matrix(Rx,     p3)                             # после Rx
    p5 = apply_matrix(R_pos,  p4)                             # после R(alpha)
    p6 = apply_matrix(T_posb, p5)                             # после T(0,b)
    p_comb = apply_matrix(M_comb, points)                     # комплексный итог

    # ----- КАРТИНКА 1: исходная фигура -----
    fig1, ax1 = plt.subplots(figsize=(8, 8))                  # новая фигура 8x8
    draw_freeman(ax1, points, code)                           # рисуем исходник
    fig1.savefig(os.path.join(output_dir, fig_name + "_1_original.png"),
                 dpi=150, bbox_inches='tight')                # сохраняем PNG
    plt.close(fig1)                                           # закрываем фигуру

    # ----- КАРТИНКА 2: все шаги вместе -----
    fig2, ax2 = plt.subplots(figsize=(10, 10))                # новая фигура 10x10
    plot_polygon(ax2, points, "Исходный", 'black')                              # исходный
    plot_polygon(ax2, p1, f"После переноса T(0,{ty})", 'green')                 # перенос
    plot_polygon(ax2, p2, f"Шаг 1: T(0,{-b})", 'olive')                         # отражение 1
    plot_polygon(ax2, p3, "Шаг 2: R(-alpha)", 'orange')                         # отражение 2
    plot_polygon(ax2, p4, "Шаг 3: Rx", 'red')                                   # отражение 3
    plot_polygon(ax2, p5, "Шаг 4: R(alpha)", 'purple')                          # отражение 4
    plot_polygon(ax2, p6, f"Шаг 5: T(0,{b})", 'brown')                          # отражение 5
    plot_polygon(ax2, p_comb, "Комплексный итог", 'blue')                       # итог
    ax2.grid(True, linestyle='--', alpha=0.5)                 # сетка
    ax2.set_aspect('equal')                                   # равный масштаб
    ax2.set_title("Пошаговые преобразования: " + fig_name)    # заголовок
    ax2.legend(loc='best', fontsize=9)                        # легенда
    fig2.savefig(os.path.join(output_dir, fig_name + "_2_steps.png"),
                 dpi=150, bbox_inches='tight')                # сохраняем PNG
    plt.close(fig2)                                           # закрываем фигуру

    # ----- КАРТИНКА 3: комплексное преобразование -----
    fig3, ax3 = plt.subplots(figsize=(8, 8))                  # новая фигура 8x8
    plot_polygon(ax3, points, "Исходный", 'black')            # исходный контур
    plot_polygon(ax3, p_comb, "Комплексное преобразование", 'blue')  # результат
    ax3.grid(True, linestyle='--', alpha=0.5)                 # сетка
    ax3.set_aspect('equal')                                   # равный масштаб
    ax3.set_title("Комплексное преобразование: " + fig_name)  # заголовок
    ax3.legend(loc='best')                                    # легенда
    ax3.text(0.02, 0.02, matrix_to_text("M_combined", M_comb),        # матрица
             transform=ax3.transAxes, va='bottom', family='monospace',
             fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    fig3.savefig(os.path.join(output_dir, fig_name + "_3_combined.png"),
                 dpi=150, bbox_inches='tight')                # сохраняем PNG
    plt.close(fig3)                                           # закрываем фигуру


# =============================================================
# ЧАСТЬ 6. ТОЧКА ВХОДА
# =============================================================

if __name__ == "__main__":
    output_dir = "results"                                    # папка с результатами
    os.makedirs(output_dir, exist_ok=True)                    # создаём её, если нет

    # Три невыпуклых 9-угольника разной формы и масштаба
    figures = [
        ("figure1", "002424650"),   # маленький зигзагообразный
        ("figure2", "002246450"),   # средний с выемкой
        ("figure3", "000224546"),   # крупный «ступенчатый»
    ]

    for fig_name, code in figures:                            # перебираем фигуры
        print("Обработка", fig_name, "с кодом", code)         # выводим прогресс
        process_figure(fig_name, code, output_dir)            # обрабатываем

    print("Готово! Изображения в папке:", output_dir)         # финальное сообщение