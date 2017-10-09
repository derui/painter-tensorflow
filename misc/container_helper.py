# Copyright 2016 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper methods for working with containers in config."""


def GenerateUserData(context):
    """Generate a Userdata for cos image given a Template context
  """

    env_list = []

    if 'dockerEnv' in context.properties:
        for key, value in context.properties['dockerEnv'].iteritems():
            env_list.append({'name': key, 'value': str(value)})

    return """
  #cloud-config

users:
- name: cloudservice

write_files:
- path: /etc/systemd/system/cloudservice.service
  permissions: 0644
  owner: root
  content: |
    [Unit]
    Description=Start a simple docker container

    [Service]
    Environment="HOME=/home/cloudservice"
    ExecStartPre=/usr/share/google/dockercfg_update.sh
    ExecStart=/usr/bin/docker run --rm -p {hostPort}:{containerPort} --name=container {image}
    ExecStop=/usr/bin/docker stop container
    ExecStopPost=/usr/bin/docker rm container

runcmd:
- systemctl daemon-reload
- systemctl start cloudservice.service
""".format(
        image=context.properties['dockerImage'],
        hostPort=context.properties['port'],
        containerPort=context.properties['containerPort'])
