import logging


def make_2d_mask(mask):
    if len(mask.shape) == 4:
        return mask.squeeze(0).squeeze(0)

    elif len(mask.shape) == 3:
        return mask.squeeze(0)

    return mask


def try_install_custom_node(custom_node_url, msg):
    try:
        import cm_global
        cm_global.try_call(api='cm.try-install-custom-node',
                           sender="Impact Pack", custom_node_url=custom_node_url, msg=msg)
    except Exception:
        logging.info(msg)
        logging.info("[Impact Pack] ComfyUI-Manager is outdated. The custom node installation feature is not available.")
