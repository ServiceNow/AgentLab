import ray

context = ray.init(address="auto", ignore_reinit_error=True)

print(context)
