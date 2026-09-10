import ast

# خواندن فایل
with open('result of pixels.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# پیدا کردن بخش آرایه JSON
start_idx = content.find('[')
end_idx = content.rfind(']') + 1

list_str = content[start_idx:end_idx]

# پارس کردن داده‌ها
data = ast.literal_eval(list_str)

# فیلتر نمونه‌های Poor (0.50 <= Dice < 0.70)
poor_samples = [item for item in data if 0.50 <= item['dice'] < 0.70]

# مرتب‌سازی بر اساس index
poor_samples.sort(key=lambda x: x['index'])

# نمایش نتایج
print(f"Total 'Poor' samples (0.50 <= Dice < 0.70): {len(poor_samples)}\n")
print(f"{'Index':<8} | {'Dice':<8} | {'IoU':<8} | {'GT Pixels':<10} | {'Pred Pixels':<10}")
print("-" * 55)
for item in poor_samples:
    print(
        f"{item['index']:<8} | {item['dice']:<8.4f} | {item['iou']:<8.4f} | {item['gt_pixels']:<10.1f} | {item['pred_pixels']:<10.1f}")

# ذخیره در فایل CSV برای تحلیل بیشتر
import csv

with open('poor_samples.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['index', 'dice', 'iou', 'gt_pixels', 'pred_pixels'])
    writer.writeheader()
    writer.writerows(poor_samples)

print(f"\n✅ نتایج در فایل 'poor_samples.csv' ذخیره شد.")
