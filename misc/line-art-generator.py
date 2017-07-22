
def GenerateConfig(context):
    """Generate YAML resource configuration for frontend
    """

    FRONTEND = context.env["deployment"] + "-frontend"
    FIREWALL = context.env["deployment"] + "-application-fw"
    APPLICATION_PORT = 80
    CONTAINER_PORT = 5000
    config = [{
        'name': FRONTEND,
        'type': 'frontend.py',
        'properties': {
            'zone': context.properties["zone"],
            'dockerImage': 'asia.gcr.io/hobby-174213/line-art-generator:latest',
            'containerImage': 'cos-stable-59-9460-73-0',
            'port': APPLICATION_PORT,
            'containerPort': CONTAINER_PORT,

            # If left out will default to 1
            'size': 1,
            # If left out will default to 1
            'maxSize': 1
        }
    }, {
        'name': FIREWALL,
        'type': 'compute.v1.firewall',
        'properties': {
            'allowed': [{
                'IPProtocol': 'TCP',
                'ports': [APPLICATION_PORT]
            }],
            'sourceRanges': ['0.0.0.0/0']
        }
    }]

    return {'resources': config}
