cd examples/ci_examples

files=$(ls $folder)
for file in $files
do
    echo "Begin to run $file"
    python $file
    rval=$?
    if [ "$rval" != 0 ]; then
        echo "Error running example $file"
        exit $rval
    fi
done
