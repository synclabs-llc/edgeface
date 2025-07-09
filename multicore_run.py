import os
import time
import tensorflow as tf
import multiprocessing

def run_model_instance(instance_id, core_ids):
    try:
        os.sched_setaffinity(0, core_ids)  # Unix/Linux only
    except AttributeError:
        print("CPU affinity setting not supported on this OS.")
    except PermissionError:
        print(f"[Instance {instance_id}] Permission denied setting affinity to {core_ids}")

    tf.config.threading.set_intra_op_parallelism_threads(len(core_ids))
    tf.config.threading.set_inter_op_parallelism_threads(1)

    model = tf.saved_model.load("models/yolo11n_tf")
    infer = model.signatures["serving_default"]
    print(f"Model instance {instance_id} loaded on cores {core_ids}", flush=True)

    dummy_input = tf.random.normal((1, 3, 640, 640))
    for i in range(100):
        preds = infer(dummy_input)["output0"]
        print(f"[Instance {instance_id}] Inference {i} done", flush=True)
        time.sleep(0.1)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # Safe on all platforms

    p1 = multiprocessing.Process(target=run_model_instance, args=(1, {0, 1}))
    p2 = multiprocessing.Process(target=run_model_instance, args=(2, {2, 3}))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
