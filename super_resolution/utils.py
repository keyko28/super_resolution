def _counter(method):
    def wrapper(*args, **kwargs):

        wrapper.calls += 1
        res = method(args, kwargs)

        return res
        
    wrapper.calls = 0

    return wrapper