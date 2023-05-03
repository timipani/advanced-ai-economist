ROOT_DIR="ai_economist/"

echo -e "\n\nRunning ISORT to sort imports ..."
isort --multi-line=3 --trailing-comma --force-grid-wrap=0 --use-parentheses --line-width=88 $ROOT_DIR

echo -e "\n\nRunning BLACK to format code ..."
black --l