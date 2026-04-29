def on_enter_state(state):
    def decorator(fn):
        fn._on_enter_state = str(state)
        return fn
    return decorator


def on_state_change(fn):
    fn._on_state_change = True
    return fn
