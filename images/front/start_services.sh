#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/app

term_handler() {
    echo "Termination signal received"
    kill -TERM "$FRONTEND_PID" 2>/dev/null
}

trap 'term_handler' SIGTERM SIGINT

uv run python3 -m streamlit run ./src/front/streamlit_app.py --server.port=6030 --server.enableXsrfProtection=false --server.baseUrlPath=/bankfcs  &
FRONTEND_PID=$!

wait -n
exit $?
