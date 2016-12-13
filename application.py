from aiohttp import web
import handlers

app = web.Application()

app.router.add_get('/blocks', handlers.get_blocks)
app.router.add_post('/blocks', handlers.post_blocks)

app.router.add_get('/inputs', handlers.get_inputs)

app.router.add_post('/train', handlers.train)
app.router.add_post('/forecast', handlers.forecast)
app.router.add_get('/consensus', handlers.consensus)

app.router.add_static('/', 'static')

web.run_app(app)
