echo "Running run_per_instance.py from run.sh";
{ time nohup python3 run_per_instance.py ; } 2> python_output.txt &
