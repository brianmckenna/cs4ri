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

    id         = data.get('id', None)
    input_keys = data.get('inputs', None)
    model_type = data.get('model', None)

    if id is None:
        return web.Response(status=400, text='id required')
    if input_keys is None:
        return web.Response(status=400, text='inputs required')
    if model_type is None:
        return web.Response(status=400, text='model required')

    res = model.train(id, model_type, input_keys)


    return web.Response(status=200)

async def forecast(request):

    data = await request.json()

    print('forecast')

    print(data)

    return web.Response(status=200)
