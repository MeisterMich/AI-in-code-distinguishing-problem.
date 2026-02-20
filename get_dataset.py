import code_tools_2 as ct
import multiprocessing as mp
from random import randint
from os import path, makedirs
from shutil import rmtree
import hashlib
from sys import argv
import time
import traceback
def matrix_to_binary_string(matrix):
    """Преобразует двоичную матрицу в строку: каждая строка матрицы как последовательность 0/1 без пробелов, строки разделяются пробелами"""
    if matrix is None:
        return ""
    rows = matrix.nrows()
    cols = matrix.ncols()
    row_strings = []
    for i in range(rows):
        row = []
        for j in range(cols):
            element = matrix[i, j]
            if hasattr(element, 'integer_representation'):
                int_value = element.integer_representation()
            else:
                int_value = int(element)
            if int_value not in (0, 1):
                int_value = int_value % 2
            row.append(str(int_value))
        row_strings.append("".join(row))
    return " ".join(row_strings)

def balancer(iters):
    tnum = mp.cpu_count()
    if iters <= 0:
        return []
    if iters < tnum:
        return [(i, i+1) for i in range(iters)]
    if not iters % tnum:
        quan = iters // tnum
        return [(i * quan, (i + 1) * quan) for i in range(tnum)]
    else:
        quan = iters // (tnum - 1)
        tr = iters - quan * (tnum - 1)
        tmp = [(i * quan, (i + 1) * quan) for i in range(tnum - 1)]
        tmp.append(((tnum - 1) * quan, (tnum - 1) * quan + tr))
        return tmp

def generator(args):
    hashes, task, counter, lock, output_file, total_needed = args
    task_start, task_end = task
    need = task_end - task_start
    made = 0
    n = 158
    k = 79

    while made < need:
        try:
            var = randint(0, 1)
            m_1 = None
            if var == 1:
                print("MDPCCodes")
                code = ct.MDPCCodes(n, k, 10)
                m_1 = code.get_generator_matrix().rref()
                # var_1 = randint(0, 3)
                # if var_1 == 0:
                #     print("AlternantCode")
                #     code = ct.AlternantCode(n, k, 512)
                #     m_1 = code.get_generator_matrix().rref()
                # elif var_1 == 1:
                #     print("GoppaCode")
                #     code = ct.GoppaCode(7, n, 2, 2)
                #     m_1 = code.get_generator_matrix().rref()
                # elif var_1 == 2:
                #     print("MDPCCodes")
                #     code = ct.MDPCCodes(n, k)
                #     m_1 = code.get_generator_matrix().rref()
                # elif var_1 == 3:
                #     print("QuasiCyclicCode")
                #     code = ct.QuasiCyclicCode(n, k, 2)
                #     m_1 = code.get_generator_matrix()
            else:
                print("GoppaCode")
                # code = ct.GoppaCode(6, n, 9, 2)
                # m_1 = code.get_generator_matrix().rref()
                m_1 = ct.generate_random_binary_matrix(k, n).rref()

            if m_1 is None:
                continue

            matr = matrix_to_binary_string(m_1)
            m4hash = hashlib.sha256(matr.encode()).hexdigest()

            with lock:
                if m4hash in hashes:
                    continue
                hashes[m4hash] = 1
                curr = counter.value
                counter.value += 1
                made += 1

                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(f"{var} {matr}\n")
                    f.flush()

            print(f"Сгенерировано {curr}/{total_needed} примеров.")

        except Exception:
            print("Ошибка при генерации, пропускаем...")
            traceback.print_exc()
            time.sleep(0.1)

            
def main():
    if len(argv) < 3:
        print("Использование: python script.py <samples_num> <output_directory> [chunk_size]")
        return

    samples_num = int(argv[1])
    output_dir = argv[2]
    chunk_size = int(argv[3]) if len(argv) >= 4 and int(argv[3]) > 0 else samples_num

    print(f"Всего примеров: {samples_num}")
    print(f"Размер одной части: {chunk_size}")
    print(f"Выходной каталог: {output_dir}")

    if path.exists(output_dir):
        rmtree(output_dir)
    makedirs(output_dir, exist_ok=True)

    manager = mp.Manager()
    hashes = manager.dict()
    counter = manager.Value('i', 0)
    lock = manager.Lock()

    num_chunks = (samples_num + chunk_size - 1) // chunk_size
    remaining = samples_num
    start_total = time.time()

    for chunk_idx in range(num_chunks):
        current_chunk_size = min(chunk_size, remaining)
        output_file = path.join(output_dir, f"part_{chunk_idx + 1}.txt")
        print(f"\n--- Часть {chunk_idx + 1} ({current_chunk_size} примеров) -> {output_file} ---")
        before = counter.value

        tasks = balancer(current_chunk_size)
        if not tasks:
            continue

        args_list = [(hashes, task, counter, lock, output_file, samples_num) for task in tasks]
        num_workers = min(mp.cpu_count(), len(tasks))
        print(f"Запуск {num_workers} процессов...")

        start_chunk = time.time()
        with mp.Pool(processes=num_workers) as pool:
            pool.map(generator, args_list)
        end_chunk = time.time()

        unique_in_chunk = counter.value - before
        remaining -= current_chunk_size
        print(f"Часть {chunk_idx + 1} завершена. Создано уникальных: {unique_in_chunk}")
        print(f"Время выполнения части: {end_chunk - start_chunk:.2f} сек")

    end_total = time.time()
    print(f"\nГенерация датасета завершена!")
    print(f"Всего уникальных образцов: {counter.value} из {samples_num} запрошенных")
    print(f"Результаты сохранены в каталоге: {output_dir}")
    print(f"Общее время выполнения: {end_total - start_total:.2f} секунд")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()