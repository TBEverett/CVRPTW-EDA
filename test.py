import time
from multiprocessing.pool import ThreadPool

# Define a simple function that simulates some work
def worker(n):
    print(f'Worker {n} started')
    for i in range(1000000000):
        continue
    print(f'Worker {n} finished')
    return f'Result from worker {n}'

# Main function to create and use the thread pool
def main():
    # Create a thread pool with a specified number of worker threads
    pool = ThreadPool(processes=3)
    
    # Start a few tasks
    tasks = [pool.apply_async(worker, (i,)) for i in range(5)]
    
    # Retrieve the results as they complete
    for task in tasks:
        result = task.get()
        print(result)
    
    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
