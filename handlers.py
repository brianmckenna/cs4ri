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
    inputs = data.get('inputs', None)
    model  = data.get('model', None)
    if not inputs:
        return web.Response(status=400, text='inputs required')
    if not model:
        return web.Response(status=400, text='model required')
    res = model.train(model, inputs)
    return web.Response(status=200)

async def forecast(request):
    data = await request.json()
    print('forecast')
    print(data)
    return web.Response(status=200)
