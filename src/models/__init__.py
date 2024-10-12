import os

# Dynamically import all model files
for module in os.listdir(os.path.dirname(__file__)):
    if module.endswith('.py') and not module.startswith('__'):
        __import__(f'{__name__}.{module[:-3]}', locals(), globals())

# Dynamically generate __all__
__all__ = [name for name in locals() if name.endswith('Model') and not name.startswith('_')]
