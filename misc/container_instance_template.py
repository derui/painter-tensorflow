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

"""Creates a Container VM with the provided Container manifest."""

from container_helper import GenerateUserData

def GenerateConfig(context):
  """Generates configuration."""

  image = ''.join(['https://www.googleapis.com/compute/v1/',
                   'projects/cos-cloud/global/images/',
                   context.properties['containerImage']])

  default_network = ''.join(['https://www.googleapis.com/compute/v1/projects/',
                             context.env['project'],
                             '/global/networks/default'])

  instance_template = {
      'name': context.env['name'] + '-it',
      'type': 'compute.v1.instanceTemplate',
      'properties': {
          'properties': {
              'metadata': {
                  'items': [{
                      'key': 'user-data',
                      'value': GenerateUserData(context)
                  }]
              },
              'machineType': 'n1-highcpu-2',
              'disks': [{
                  'deviceName': 'boot',
                  'boot': True,
                  'autoDelete': True,
                  'mode': 'READ_WRITE',
                  'type': 'PERSISTENT',
                  'initializeParams': {'sourceImage': image}
              }],
              'scheduling': {
                  'preemptible': True
              },
              'serviceAccounts': [
                  {
                      'email': '874188143542-compute@developer.gserviceaccount.com',
                      'scopes': ['https://www.googleapis.com/auth/cloud-platform.read-only']
                  }
              ],
              'networkInterfaces': [{
                  'network': default_network
              }]
          }
      }
  }

  return {'resources': [instance_template]}