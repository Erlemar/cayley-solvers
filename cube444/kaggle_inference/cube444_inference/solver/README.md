# 4x4x4 full-cube ensemble inference

Самодостаточный inference-бандл для полного решения 4x4x4 и улучшения Kaggle submission.

Ансамбль зафиксирован по лучшему измеренному варианту:

```text
Q = 0.60 * Q_transformer + 0.40 * Q_x16_MLP
```

На 30 одинаковых случайных скрамблах при `B=2^18` он дал среднюю длину `50.93`
против `53.00` у transformer baseline; парная разница `-2.07` хода,
95% Student t CI `[-3.23, -0.90]`.

## Установка

Нужны Python 3.10+, PyTorch с CUDA, а также зависимости:

```bash
python3 -m pip install -r requirements.txt
```

PyTorch лучше установить отдельно под CUDA кластера с https://pytorch.org/.

Проверка архива без загрузки моделей на GPU и без поиска:

```bash
python3 solve_ensemble_submission.py --dry-run
```

Она проверяет SHA весов, генераторы, target и все пути в `baseline_submission.csv`.
Встроенный baseline — актуальный проверенный union длины `47 588`; оригинальный
`sample_submission.csv` также лежит рядом для справки.

## Production-прогон

По умолчанию скрипт запускает `B=2^21`, 150 шагов и две попытки:

```bash
./run_submission.sh 2>&1 | tee results/run.log
```

Значения можно менять окружением, например для `B=2^18`:

```bash
BEAM_SIZE=262144 NUM_STEPS=150 NUM_ATTEMPTS=2 ./run_submission.sh \
  2>&1 | tee results/run-B262144.log
```

После каждого куба runner:

1. точно проигрывает найденную последовательность и сверяет с собранным target;
2. коммитит результат в `results/progress-shard0.sqlite3`;
3. атомарно обновляет CSV и JSON-report;
4. оставляет для каждого ID более короткий из нового решения и включённого baseline.

Повтор той же команды безопасно продолжает прогон. Уже обработанные теми же моделями и
параметрами состояния пропускаются. Для повторной попытки не найденных состояний добавьте
`--retry-failed` в прямой вызов Python. Для сознательного перепоиска уже решённых —
`--rerun-solved`.

## Несколько GPU/нод

На каждой ноде задайте одинаковое число шардов и свой индекс:

```bash
NUM_SHARDS=8 SHARD_INDEX=0 ./run_submission.sh 2>&1 | tee results/run-shard0.log
NUM_SHARDS=8 SHARD_INDEX=1 ./run_submission.sh 2>&1 | tee results/run-shard1.log
```

Не давайте разным нодам один SQLite-файл по сетевой файловой системе. Верните отдельные
`progress-shardN.sqlite3`, затем объедините их:

```bash
python3 merge_ensemble_progress.py results/progress-shard*.sqlite3 \
  --output results/submission-merged.csv
```

Merge повторно валидирует каждую последовательность и выбирает кратчайшую.

## Что вернуть после вычислений

Достаточно упаковать базу, финальный CSV, report и лог:

```bash
tar -czf cube4-ensemble-results.tar.gz \
  results/*.sqlite3 results/*.csv results/*.json results/*.log
```

Главный готовый Kaggle-файл — `results/submission-shard0.csv` для одной ноды или
`results/submission-merged.csv` после шардированного запуска. Благодаря baseline он остаётся
полным и валидным даже когда ансамбль не нашёл решение для части состояний.

## Быстрый smoke test

На GPU можно проверить один пример маленьким лучом:

```bash
python3 solve_ensemble_submission.py \
  --limit 1 --B 16384 --num-steps 150 --num-attempts 1 \
  --progress-db results/smoke.sqlite3 \
  --output results/smoke.csv
```

Большой луч дорогой. Не начинайте production-прогон, пока `--dry-run` и smoke test не
завершились без ошибок.
