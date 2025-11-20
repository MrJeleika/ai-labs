"""
Скрипт для запуску всіх лабораторних робіт послідовно
"""

import os
import sys

def run_task(filename):
    """Запускає окреме завдання"""
    print("\n" + "=" * 80)
    print(f"ЗАПУСК: {filename}")
    print("=" * 80)
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            code = f.read()
        exec(code)
        print(f"\n✓ {filename} виконано успішно")
    except Exception as e:
        print(f"\n✗ Помилка при виконанні {filename}: {e}")
        return False
    return True

def main():
    """Головна функція"""
    tasks = [
        'LR_7_task_1.py',
        'LR_7_task_2.py',
        'LR_7_task_3.py',
        'LR_8_task_4.py',
        'LR_8_task_2.py'
    ]
    
    print("=" * 80)
    print("ЛАБОРАТОРНІ РОБОТИ 7-8: Лінійна та поліноміальна регресія")
    print("=" * 80)
    print(f"Кількість завдань: {len(tasks)}")
    print()
    
    input("Натисніть Enter для початку виконання всіх завдань...")
    
    results = {}
    for task in tasks:
        if os.path.exists(task):
            results[task] = run_task(task)
        else:
            print(f"\n✗ Файл {task} не знайдено")
            results[task] = False
    
    # Підсумок
    print("\n" + "=" * 80)
    print("ПІДСУМОК ВИКОНАННЯ:")
    print("=" * 80)
    for task, success in results.items():
        status = "✓ УСПІШНО" if success else "✗ ПОМИЛКА"
        print(f"{status}: {task}")
    
    successful = sum(results.values())
    total = len(results)
    print(f"\nВиконано успішно: {successful}/{total}")
    print("=" * 80)

if __name__ == "__main__":
    main()

