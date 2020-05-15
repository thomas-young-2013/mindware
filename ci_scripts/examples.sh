echo '###############################################################################'
echo '###############################################################################'
echo "Starting to test estimator api"
echo '###############################################################################'
cd branin
python branin_fmin.py
rval=$?
if [ "$rval" != 0 ]; then
    echo "Error running example branin_fmin.py"
    exit $rval
fi
