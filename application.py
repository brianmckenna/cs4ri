from aiohttp import web
import aiohttp_jinja2
import handlers
import jinja2
import model

app = web.Application()

aiohttp_jinja2.setup(app,loader=jinja2.FileSystemLoader('templates'))


app.router.add_get('/blocks', handlers.get_blocks)
app.router.add_post('/blocks', handlers.post_blocks)

app.router.add_get('/inputs', handlers.get_inputs)

app.router.add_post('/train', handlers.train)
app.router.add_post('/forecast', handlers.forecast)
app.router.add_get('/consensus', handlers.consensus)
app.router.add_get('/results', handlers.results)

app.router.add_static('/', 'static')



async def start_background_tasks(app):
    app['update_obs'] = app.loop.create_task(model.update_obs())

app.on_startup.append(start_background_tasks)

web.run_app(app)
