for f in texts/*
do
    name=`python -c "import os; print(os.path.basename('$f').split('.')[0])"`
    python train.py $name
done