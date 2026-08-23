import os

# پوشه‌هایی که می‌خواهیم کاملاً نادیده بگیریم
IGNORE_DIRS = {'.venv', 'venv', 'env', '__pycache__', '.git', 'node_modules', 'build', 'dist', 'logs', 'temp'}


def get_filtered_items(path):
    """لیست مرتبی از پوشه‌ها و فایل‌های مورد نظر را برمی‌گرداند."""
    try:
        items = os.listdir(path)
    except PermissionError:
        return [], [], [], []

    dirs = []
    py_files = []
    pth_files = []
    png_files = []

    for item in items:
        full = os.path.join(path, item)
        if os.path.isdir(full):
            if item not in IGNORE_DIRS and not item.startswith('.'):
                dirs.append(item)
        else:
            if item.endswith('.py'):
                py_files.append(item)
            elif item.endswith('.pth'):
                pth_files.append(item)
            elif item.endswith('.png'):
                png_files.append(item)

    # مرتب‌سازی بر اساس نام
    dirs.sort()
    py_files.sort()
    pth_files.sort()
    png_files.sort()

    # فقط ۱۰ فایل PNG اول را نگه می‌داریم
    png_files = png_files[:10]

    return dirs, py_files, pth_files, png_files


def print_structure(path, indent='', is_last=True):
    """چاپ ساختار درختی با نمایش همه پوشه‌ها، فایل‌های .py، .pth و ۱۰ فایل PNG اول."""
    folder_name = os.path.basename(path)
    prefix = '└── ' if is_last else '├── '
    print(indent + prefix + folder_name + '/')

    new_indent = indent + ('    ' if is_last else '│   ')

    dirs, py_files, pth_files, png_files = get_filtered_items(path)

    # نمایش فایل‌های .py و .pth با هم
    all_code_files = py_files + pth_files
    for idx, file in enumerate(all_code_files):
        is_file_last = (idx == len(all_code_files) - 1 and len(png_files) == 0 and len(dirs) == 0)
        file_prefix = '└── ' if is_file_last else '├── '
        print(new_indent + file_prefix + file)

    # نمایش فایل‌های .png (حداکثر ۱۰ تا)
    for idx, file in enumerate(png_files):
        is_png_last = (idx == len(png_files) - 1 and len(dirs) == 0)
        png_prefix = '└── ' if is_png_last else '├── '
        print(new_indent + png_prefix + file + ' (PNG)')

    # نمایش زیرپوشه‌ها (حتی اگر فایل پایتون نداشته باشند)
    for idx, dir_name in enumerate(dirs):
        full_dir = os.path.join(path, dir_name)
        is_last_dir = (idx == len(dirs) - 1)
        print_structure(full_dir, new_indent, is_last_dir)


if __name__ == "__main__":
    root = os.getcwd()
    print(os.path.basename(root) + '/')
    print_structure(root, '', True)
