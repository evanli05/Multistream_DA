import logging

logging.basicConfig(filename = 'testLogger.log', level = logging.DEBUG)

logger = logging.getLogger()

logger.info('Our first logger')