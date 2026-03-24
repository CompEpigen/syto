NUMBA_DISABLE_JIT=1 poetry run coverage run
poetry run coverage report
poetry run coverage xml
poetry run genbadge coverage -i coverage.xml
rm coverage.xml