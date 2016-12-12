from aiohttp import web

import model

async def get_blocks(request):
    with open('blocks.xml', 'r') as f:
        xml = f.read()
    return web.Response(text=xml, status=200)

async def post_blocks(request):
    data = await request.post()
    with open('blocks.xml', 'w') as f:
        f.write(data['xml'])
    return web.Response(status=200)

async def train(request):

    data = await request.json()

    cs4ri_id   = data.get('cs4ri_id', None)
    input_keys = data.get('inputs', None)
    model_type = data.get('model', None)

    if not cs4ri_id:
        return web.Response(status=400, text='CS4RI id required')
    if input_keys is None:
        return web.Response(status=400, text='inputs required')
    if model_type is None:
        return web.Response(status=400, text='model required')

    res = model.train(cs4ri_id, model_type, input_keys)

    return web.Response(body=res, status=200)

async def forecast(request):

    data = await request.json()

    cs4ri_id   = data.get('cs4ri_id', None)

    if cs4ri_id is None:
        return web.Response(status=400, text='CS4RI id required')

    res = model.forecast(cs4ri_id)

    return web.Response(body=res, status=200)
