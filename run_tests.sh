NUMBA_DISABLE_JIT=1 poetry run coverage run
poetry run coverage report
poetry run coverage xml
poetry run genbadge coverage -i coverage.xml
mv coverage-badge.svg assets/coverage-badge.svg
rm coverage.xml