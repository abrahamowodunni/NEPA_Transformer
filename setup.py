from setuptools import find_packages, setup
from typing import List

E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function will return the list of requirements,
    removing any invalid specifiers like '-e .'
    '''
    requirements = []
    with open(file_path, 'r') as file_obj:
        # Read all lines from the requirements file
        requirements = file_obj.readlines()
        # Strip newline characters and clean entries
        requirements = [req.strip() for req in requirements if req.strip()]
        
        # Remove '-e .' if it exists in the list
        if E_DOT in requirements:
            requirements.remove(E_DOT)
    
    # Print to debug if '-e .' was correctly removed
    print("Final requirements:", requirements)
    
    return requirements

setup(
    name='NEPA_Transformer',
    version='0.0.0.0',
    author='Abraham Owodunni',
    author_email='abrahamowodunni@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')  # Retrieve requirements
)
