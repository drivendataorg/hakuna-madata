#!/bin/bash

: '
Использование:
`./lint.sh <команда> <путь к директории с кодом>`,
например `./lint.sh format .`

<путь к директории> с кодом по умолчанию равен текущей папке, так что можно запускать и так из корневной директории проекта:
`./lint.sh format`

Из-за особенностей работы isort с устанавливаемыми зависимостями, он по умолчанию выключен.
Для проверки/фрматирования с использованием isort можно воспользоваться командами check-local/format-local.
Перед этим необходимо убедиться что все зависимости проекта установлены в локальном окружении.

Команды:

install - установить в текущее окружение библиотеки для форматирования и линтинга

check - проверить, что код проходит проверку линтером и форматером
check-local - аналог check, с добавленной проеверкой на сортировку импортов

format - отформатировать весь код
format-local - аналог format, с добавленной проеверкой на сортировку импортов

check-black - проверка форматирования
check-isort - проверка сортировки импортов
check-flake8 - проверка линтером
check-mypy - проверка типизации mypy

diff-isort - вывод diff для сортировщика импортов
diff-black - вывод diff для форматтера
'

SCRIPT_NAME=$0
COMMAND=$1
FILEPATH=${2:-.}

function handle_exit {
    EXIT_CODE=$1
    if [[ $EXIT_CODE -ne 0 ]]; then
        echo $2
    fi
    exit $EXIT_CODE
}

function find_python_projects {
    # Find subdirectories with at least one __init__.py file inside

    shopt -s nullglob
    DIRS=($FILEPATH/*/)
    shopt -u nullglob # Turn off nullglob to make sure it doesn't interfere with anything later

    for DIR in "${DIRS[@]}"; do
        PY_FILES_NUM=$(find "$DIR" -name "__init__.py" | wc -l)

        if [[ $PY_FILES_NUM -gt 0 ]]; then
            echo "$DIR"
        fi
    done
}

case ${COMMAND} in
    check-black)
        black --config ./black.toml --check ${FILEPATH}
        handle_exit $? "Formatting error! Run \`$SCRIPT_NAME format\` to format the code"
        ;;

    check-isort)
        isort -rc ${FILEPATH} --check-only
        handle_exit $? "Isort error! Run \`$SCRIPT_NAME format\` to format the code"
        ;;

    check-flake8)
        flake8 ${FILEPATH}
        handle_exit $? "Flake8 error!"
        ;;

    check-mypy)
        find_python_projects | xargs -i mypy {}
        handle_exit $? "Mypy error!"
        ;;

    check)
        set -e
        $SCRIPT_NAME check-black $FILEPATH
        $SCRIPT_NAME check-flake8 $FILEPATH
        $SCRIPT_NAME check-mypy $FILEPATH
        ;;

    check-local)
        set -e
        $SCRIPT_NAME check-isort $FILEPATH
        $SCRIPT_NAME check-black $FILEPATH
        $SCRIPT_NAME check-flake8 $FILEPATH
        $SCRIPT_NAME check-mypy $FILEPATH
        ;;

    diff-isort)
        isort -rc ${FILEPATH} --diff
        ;;

    diff-black)
        black --config "$FILEPATH/black.toml" --diff ${FILEPATH}
        ;;

    format)
        black --config ./black.toml ${FILEPATH}
        ;;

    format-local)
        isort -rc ${FILEPATH}
        black --config ./black.toml ${FILEPATH}
        ;;

    install)
        pip install black flake8 isort mypy
        ;;
    *)
        echo $"Usage: $SCRIPT_NAME {check|check-local|check-black|check-isort|check-flake8|check-mypy|diff-isort|diff-black|format|format-local|install}"
        exit 1
esac
