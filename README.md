# DBSCAN Visualization with Pygame

Проект реализует визуализацию алгоритма кластеризации DBSCAN с использованием библиотеки Pygame для отображения процесса обработки данных в режиме реального времени.

## Функциональность
- Визуализация кластеризации DBSCAN для двухмерных точек.
- Интерактивное добавление точек с помощью мыши.
- Показ шума, граничных и центральных точек кластера с разными цветами.
- Отображение кластеров с помощью библиотеки Matplotlib.

## Используемые технологии
- **Python**
- **Pygame** для визуализации
- **Scikit-learn** для применения алгоритма DBSCAN
- **Matplotlib** для построения графиков
- 
## Управление
- **Левая кнопка мыши**: Добавление точек на поле.
- **Пробел**: Вызов DBSCAN из Scikit-learn и отображение результатов через Matplotlib.
- **Enter**: Применение кастомного алгоритма DBSCAN с визуализацией результатов.
- **Escape**: Очистка экрана и отображение кластеров разными цветами.
- **Закрытие окна**: Завершение программы.

## Особенности реализации
**Алгоритм DBSCAN:**
  Реализация DBSCAN из Scikit-learn.
  Кастомная версия DBSCAN с выделением центральных (зеленый), граничных (желтый) и шумовых (красный) точек.
**Генерация точек:**
  Добавление нескольких соседних точек при каждом клике мыши для имитации плотных регионов.

