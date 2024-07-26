import json


def auto_cast(val):
    # return float(val) if val.isdigit() else bool(val)
    try:
        return json.loads(val)
    except:
        return bool(val)


def to_params_str(params_dict):
    str_params = ''
    for k, v in params_dict.items():
        if len(str_params) > 0:
            str_params += ':'

        if isinstance(v, bool) and True:
            str_params += f'{k}'
        else:
            str_params += f'{k}={v}'
    return str_params


def from_params_str(params_list, index):
    """
    Parse optimizers/schedulers parameters string
    :param params_list: list of strings from CLI
    :param index: index of params string in params_list
    :return:
    """
    args = []
    kwargs = {}

    if params_list is not None:
        params = params_list[index]
        for p in params.split(':'):
            p = p.split('=')
            if len(p) == 2:
                kwargs[p[0]] = auto_cast(p[1])
            else:
                args.append(auto_cast(p[0]))

    return args, kwargs
