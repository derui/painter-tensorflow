# encoding: utf8

import base64
import os
from PIL import Image
import io
import argparse
import numpy as np
from werkzeug.utils import redirect
from werkzeug.wrappers import Response, Request
from werkzeug.routing import Map, Rule
from werkzeug.exceptions import HTTPException, NotFound
from werkzeug.wsgi import SharedDataMiddleware, wrap_file


class LineArtConverter(object):
    def __init__(self, config):
        self.config = config
        self.url_map = Map([
            Rule('/', endpoint='index'),
            Rule('/api/generate_image', endpoint='generate_image', methods=["POST"])
        ])

    def dispatch_request(self, request):
        adapter = self.url_map.bind_to_environ(request.environ)
        try:
            endpoint, values = adapter.match()
            return getattr(self, 'on_' + endpoint)(request, **values)
        except HTTPException as e:
            return e

    def on_index(self, request):
        return redirect('/static/index.html')

    def on_generate_image(self, request):
        image = base64.b64decode(request.form['image'])

        image = Image.open(io.BytesIO(image)).convert('RGB')
        img_array = np.asarray(image)
        img_array = img_array.astype(np.float32)
        img_array = np.multiply(img_array, 1 / 255.0)

        img = self.config['generate'](img_array)

        img = np.multiply(img, 255.0)
        img = img.astype(np.uint8)

        image = Image.fromarray(img.reshape([512, 512]), 'L')
        ret = io.BytesIO()
        image.save(ret, format='png')
        ret.seek(0)
        return Response(ret.getvalue(), mimetype="image/png")

    def wsgi_app(self, environ, start_response):
        request = Request(environ)
        response = self.dispatch_request(request)
        return response(environ, start_response)

    def __call__(self, environ, start_response):
        return self.wsgi_app(environ, start_response)


def create_app(generate_func, with_static=True):
    app = LineArtConverter({'generate': generate_func})

    if with_static:
        app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
            '/static': os.path.join(os.path.dirname(__file__), 'static')
        })

    return app


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Serve api')
    argparser.add_argument(
        '--train_dir',
        default='./log',
        type=str,
        help='Directory will have been saving checkpoint')

    ARGS = argparser.parse_args()

    from werkzeug.serving import run_simple
    import line_art_generator.lib.generator as generator

    sess, op, x = generator.init_sess(1, 512, 512, ARGS.train_dir)

    def generate(image):
        [ret] = generator.generate(sess, op, x, [image])
        return ret

    app = create_app(generate)
    run_simple('127.0.0.1', 5000, app, use_debugger=True)
