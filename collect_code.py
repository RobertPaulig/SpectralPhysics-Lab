"""
Скрипт для сбора всего кода проекта в один текстовый файл.
Исключает системные файлы и директории.
"""

import os
from pathlib import Path

# Директории и файлы для исключения
EXCLUDE_DIRS = {
    '.git', '.venv', '__pycache__', '.pytest_cache', 
    'node_modules', '.idea', '.vscode', 'dist', 'build',
    '*.egg-info'
}

EXCLUDE_FILES = {
    '.gitignore', '.DS_Store', '*.pyc', '*.pyo', 
    '*.pyd', '.Python', 'pip-log.txt', '*.so',
    'collect_code.py',  # Исключаем сам этот скрипт
    'test_new_features.py',  # Старый тестовый файл
    'test_new_architecture.py',  # Тестовый файл
}

# Расширения файлов для включения
INCLUDE_EXTENSIONS = {
    '.py', '.md', '.txt', '.yml', '.yaml', 
    '.toml', '.cfg', '.ini', '.json'
}


def should_exclude_dir(dir_name):
    """Проверяет, нужно ли исключить директорию."""
    return dir_name in EXCLUDE_DIRS or dir_name.startswith('.')


def should_exclude_file(file_name):
    """Проверяет, нужно ли исключить файл."""
    if file_name in EXCLUDE_FILES:
        return True
    if file_name.startswith('.'):
        return True
    # Проверяем паттерны
    for pattern in EXCLUDE_FILES:
        if '*' in pattern:
            ext = pattern.replace('*', '')
            if file_name.endswith(ext):
                return True
    return False


def should_include_file(file_path):
    """Проверяет, нужно ли включить файл."""
    ext = file_path.suffix
    return ext in INCLUDE_EXTENSIONS


def get_tree_structure(root_path, prefix='', is_last=True):
    """Генерирует древовидную структуру директорий."""
    lines = []
    root = Path(root_path)
    
    if root.is_file():
        return lines
    
    try:
        items = sorted(root.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        items = [item for item in items if not should_exclude_dir(item.name) 
                 and not should_exclude_file(item.name)]
        
        for i, item in enumerate(items):
            is_last_item = (i == len(items) - 1)
            
            # Символы для отрисовки дерева
            connector = '└── ' if is_last_item else '├── '
            lines.append(f'{prefix}{connector}{item.name}')
            
            if item.is_dir():
                extension = '    ' if is_last_item else '│   '
                lines.extend(get_tree_structure(item, prefix + extension, is_last_item))
    
    except PermissionError:
        pass
    
    return lines


def collect_code_files(root_path):
    """Собирает все файлы с кодом."""
    files_content = []
    root = Path(root_path)
    
    for file_path in sorted(root.rglob('*')):
        # Пропускаем директории
        if file_path.is_dir():
            continue
        
        # Проверяем, находится ли файл в исключаемой директории
        if any(should_exclude_dir(part) for part in file_path.parts):
            continue
        
        # Проверяем, не исключен ли сам файл
        if should_exclude_file(file_path.name):
            continue
        
        # Проверяем расширение
        if not should_include_file(file_path):
            continue
        
        # Читаем содержимое
        try:
            relative_path = file_path.relative_to(root)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            files_content.append({
                'path': str(relative_path),
                'content': content
            })
        except Exception as e:
            print(f"Ошибка при чтении {file_path}: {e}")
    
    return files_content


def main():
    """Основная функция."""
    project_root = Path(__file__).parent
    output_file = project_root / 'PROJECT_CODE.txt'
    
    print("Собираю код проекта...")
    print(f"Корневая директория: {project_root}")
    
    # Собираем структуру дерева
    print("\nГенерирую дерево файлов...")
    tree_lines = [project_root.name]
    tree_lines.extend(get_tree_structure(project_root))
    
    # Собираем содержимое файлов
    print("Собираю содержимое файлов...")
    files = collect_code_files(project_root)
    
    # Записываем в выходной файл
    print(f"\nЗаписываю в {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        # Заголовок
        f.write("=" * 80 + "\n")
        f.write("ПРОЕКТ: SpectralPhysics-Lab\n")
        f.write("Дата создания архива: " + 
                __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("=" * 80 + "\n\n")
        
        # Дерево структуры
        f.write("СТРУКТУРА ПРОЕКТА:\n")
        f.write("-" * 80 + "\n")
        f.write('\n'.join(tree_lines))
        f.write("\n\n")
        
        # Содержимое файлов
        f.write("=" * 80 + "\n")
        f.write("СОДЕРЖИМОЕ ФАЙЛОВ\n")
        f.write("=" * 80 + "\n\n")
        
        for file_info in files:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"ФАЙЛ: {file_info['path']}\n")
            f.write("=" * 80 + "\n\n")
            f.write(file_info['content'])
            f.write("\n\n")
    
    print(f"\n✅ Готово! Собрано файлов: {len(files)}")
    print(f"📄 Результат сохранен в: {output_file}")
    print(f"📊 Размер файла: {output_file.stat().st_size / 1024:.2f} KB")


if __name__ == '__main__':
    main()
