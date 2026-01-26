import os
import sys

# --- ВАЖНО: Эти настройки должны быть ДО импорта open3d ---
# Принудительно используем программный рендеринг, чтобы избежать EGL Crash
os.environ['OPEN3D_CPU_RENDERING'] = 'true'
# Дополнительно для Linux серверов (Mesa drivers)
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1' 

import py360convert 
import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt

class SceneRenderer:
    def __init__(self, geometry_path, width=512, height=512, point_size=3.0):
        self.width = width
        self.height = height
        
        # 1. Попытка загрузить как Mesh
        self.geometry = o3d.io.read_triangle_mesh(geometry_path)
        self.is_mesh = True
        
        # Проверка: если нет треугольников, это Облако Точек
        if len(self.geometry.triangles) == 0:
            print(f"[INFO] Файл {geometry_path} не содержит треугольников. Загружаем как PointCloud.")
            self.geometry = o3d.io.read_point_cloud(geometry_path)
            self.is_mesh = False
        else:
            self.geometry.compute_vertex_normals()

        # Инициализация рендерера
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        
        # Настройка материала
        self.material = o3d.visualization.rendering.MaterialRecord()
        if self.is_mesh:
            self.material.shader = "defaultLit"
        else:
            # Для точек нужен специальный шейдер и настройка размера
            self.material.shader = "defaultUnlit" 
            self.material.point_size = point_size # Увеличьте, если сцена выглядит "дырявой"

        self.renderer.scene.add_geometry("scene", self.geometry, self.material)
        
        # Цвет фона (черный по умолчанию, можно изменить на белый для отладки)
        self.renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])

    def render_view(self, eye, center, up, fov=60.0):
        self.renderer.scene.camera.look_at(
            np.array(center, dtype=np.float32),
            np.array(eye, dtype=np.float32),
            np.array(up, dtype=np.float32)
        )
        self.renderer.scene.camera.set_projection(
            fov, 
            self.width / self.height, 
            0.1, 
            1000.0, 
            o3d.visualization.rendering.Camera.FovType.Vertical
        )
        
        img = self.renderer.render_to_image()
        return np.asarray(img)
    
    def render_cubemap(self, center_pos=[0, 0, 0]):
        """
        Рендерит 6 граней куба из центральной точки для создания панорамы.
        Порядок: Front, Right, Back, Left, Top, Bottom
        """
        fov = 90.0 # Для кубической карты FOV всегда 90 градусов
        # Направления для граней куба (look_at, up)
        # В Open3D координаты обычно Y-up или Z-up, зависит от вашей модели.
        # Здесь предполагается стандартная OpenGL система: -Z вперед, +Y вверх.
        
        # Формат: (target_vector, up_vector)
        # center_pos добавляется к target_vector при вызове
        views = [
            ([0, 0, -1], [0, 1, 0]),  # Front
            ([1, 0, 0],  [0, 1, 0]),  # Right
            ([0, 0, 1],  [0, 1, 0]),  # Back
            ([-1, 0, 0], [0, 1, 0]),  # Left
            ([0, 1, 0],  [0, 0, 1]),  # Top (Up vector меняется)
            ([0, -1, 0], [0, 0, -1]), # Bottom
        ]
        
        faces = []
        # Сохраняем текущие настройки, чтобы временно изменить разрешение на квадратное
        original_w, original_h = self.width, self.height
        # Для кубической карты грани должны быть квадратными
        cube_size = max(original_w, original_h) 
        
        # Нам нужно пересоздать рендерер для изменения размера (особенность Open3D)
        # Но для простоты примера мы просто будем надеяться, что init был с квадратным размером
        # или масштабируем результат. Для продакшена лучше пересоздать OffscreenRenderer.
        
        for look_vec, up_vec in views:
            target = np.array(center_pos) + np.array(look_vec)
            img = self.render_view(center_pos, target, up_vec, fov)
            faces.append(img)
            
        return faces


# def render(mesh_file, output_path):
# --- Использование ---
mesh_file = "/mnt/e/3D/experiments/output/WorldGen/splat.ply" # Ваш файл (часто .ply или .obj)
output_path = "/mnt/e/3D/experiments/output/WorldGen"

try:
    # point_size - важный параметр для WorldGen. 
    # Так как это облако точек, при малом размере сцена будет прозрачной. 
    # Попробуйте значения 2.0, 3.0 или 5.0.
    renderer = SceneRenderer(mesh_file, width=512, height=512, point_size=1)

    # Рендерим вид с центра (0,0,0) вперед по оси Z
    image_view = renderer.render_view(
        eye=[0, 0, 0],
        center=[0, 0, 1], # Или [0, 0, -1], зависит от осей WorldGen
        up=[0, 1, 0]
    )

    cube_faces = renderer.render_cubemap(center_pos=[0,0,0])
    for i, img_cube in enumerate(cube_faces):
        path = os.path.join(output_path, f"face_{i}_render.png")
        cv2.imwrite(path, cv2.cvtColor(img_cube, cv2.COLOR_RGB2BGR))

    equi_img = py360convert.c2e(cube_faces, 2048, 4096, cube_format='list')
    pano_path = os.path.join(output_path, f"pano_render.png")
    cv2.imwrite(pano_path, cv2.cvtColor(equi_img, cv2.COLOR_RGB2BGR))
    

    # Проверка вывода
    if image_view is None or image_view.size == 0:
        print("Ошибка: Получено пустое изображение.")
    else:
        print(f"Успешный рендер! Размер: {image_view.shape}")
        # Сохранение
        img_path = os.path.join(output_path, f"img_render.png")
        cv2.imwrite(img_path, cv2.cvtColor(image_view, cv2.COLOR_RGB2BGR))
        print(f"Сохранено в {img_path}")

except Exception as e:
    print(f"Критическая ошибка: {e}")