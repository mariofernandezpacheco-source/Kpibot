#!/usr/bin/env python3
"""
Script de migración para convertir print() statements a logging estructurado
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Tuple
import argparse


class PrintToLoggerTransformer(ast.NodeTransformer):
    """AST Transformer que convierte print() calls a logger calls"""

    def __init__(self):
        self.changes = []
        self.has_logger_import = False
        self.logger_name = None

    def visit_Module(self, node):
        """Visita el módulo para añadir imports si es necesario"""

        # Buscar imports existentes
        for item in node.body:
            if isinstance(item, ast.Import):
                for alias in item.names:
                    if alias.name == 'logging':
                        self.has_logger_import = True
            elif isinstance(item, ast.ImportFrom):
                if item.module == 'utils.logging_enhanced':
                    self.has_logger_import = True

        # Transformar el resto del módulo
        self.generic_visit(node)

        # Añadir imports si es necesario
        if self.changes and not self.has_logger_import:
            # Crear import para logging estructurado
            import_node = ast.ImportFrom(
                module='utils.logging_enhanced',
                names=[ast.alias(name='get_logger', asname=None)],
                level=0
            )

            # Insertar después de otros imports o al principio
            insert_pos = 0
            for i, item in enumerate(node.body):
                if isinstance(item, (ast.Import, ast.ImportFrom)):
                    insert_pos = i + 1
                else:
                    break

            node.body.insert(insert_pos, import_node)

            # Crear logger variable
            logger_assign = ast.Assign(
                targets=[ast.Name(id='logger', ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id='get_logger', ctx=ast.Load()),
                    args=[ast.Constant(value='__name__')],
                    keywords=[]
                )
            )
            node.body.insert(insert_pos + 1, logger_assign)

        return node

    def visit_Call(self, node):
        """Visita llamadas a funciones para encontrar print()"""

        # Transformar recursivamente argumentos
        self.generic_visit(node)

        # Buscar llamadas a print()
        if (isinstance(node.func, ast.Name) and
                node.func.id == 'print'):
            return self._transform_print_to_logger(node)

        return node

    def _transform_print_to_logger(self, print_node):
        """Transforma una llamada print() a logger call"""

        # Analizar argumentos del print
        if not print_node.args:
            # print() sin argumentos -> logger.info("")
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='logger', ctx=ast.Load()),
                    attr='info',
                    ctx=ast.Load()
                ),
                args=[ast.Constant(value='empty_print'), ast.Constant(value="")],
                keywords=[]
            )

        # Obtener primer argumento como mensaje principal
        first_arg = print_node.args[0]

        # Intentar extraer event name del mensaje
        event_name = "info_message"
        message = ""

        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            message = first_arg.value
            event_name = self._extract_event_name(message)
        elif isinstance(first_arg, ast.JoinedStr):  # f-string
            # Para f-strings, usar un event genérico
            event_name = "formatted_message"
            # Mantener la f-string como mensaje

        # Crear llamada al logger
        logger_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='logger', ctx=ast.Load()),
                attr='info',  # Por defecto usar info
                ctx=ast.Load()
            ),
            args=[
                ast.Constant(value=event_name),  # event
                first_arg  # message (mantener formato original)
            ],
            keywords=[]
        )

        # Si hay argumentos adicionales, convertir a keywords
        if len(print_node.args) > 1:
            for i, arg in enumerate(print_node.args[1:], 1):
                logger_call.keywords.append(
                    ast.keyword(
                        arg=f'arg_{i}',
                        value=arg
                    )
                )

        self.changes.append({
            'type': 'print_to_logger',
            'event': event_name,
            'original': ast.unparse(print_node),
            'transformed': ast.unparse(logger_call)
        })

        return logger_call

    def _extract_event_name(self, message: str) -> str:
        """Extrae un event name razonable del mensaje"""

        # Patrones comunes para detectar eventos
        patterns = [
            (r'error', 'error'),
            (r'warning|warn', 'warning'),
            (r'success|ok|completed', 'success'),
            (r'starting|iniciando|begin', 'start'),
            (r'finished|terminado|done', 'finished'),
            (r'loading|cargando', 'loading'),
            (r'saving|guardando', 'saving'),
            (r'connecting|conectando', 'connecting'),
            (r'processing|procesando', 'processing'),
            (r'training|entrenando', 'training'),
            (r'downloading|descargando', 'downloading'),
        ]

        message_lower = message.lower()

        for pattern, event in patterns:
            if re.search(pattern, message_lower):
                return event

        # Si no encuentra patrón específico, crear event genérico
        # Tomar las primeras palabras significativas
        words = re.findall(r'\w+', message_lower)
        if words:
            # Tomar hasta 3 palabras, filtrar palabras comunes
            stop_words = {'el', 'la', 'de', 'en', 'un', 'una', 'y', 'o', 'the', 'a', 'an', 'and', 'or', 'is', 'are'}
            significant_words = [w for w in words[:3] if w not in stop_words]
            if significant_words:
                return '_'.join(significant_words)

        return 'info_message'


def analyze_python_file(file_path: Path) -> Dict:
    """Analiza un archivo Python y encuentra print() statements"""

    try:
        content = file_path.read_text(encoding='utf-8')
        tree = ast.parse(content)

        # Contar prints existentes
        print_count = 0
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and
                    isinstance(node.func, ast.Name) and
                    node.func.id == 'print'):
                print_count += 1

        return {
            'file': file_path,
            'print_count': print_count,
            'parseable': True,
            'content': content,
            'tree': tree
        }

    except Exception as e:
        return {
            'file': file_path,
            'print_count': 0,
            'parseable': False,
            'error': str(e)
        }


def transform_file(file_info: Dict) -> Dict:
    """Transforma un archivo, convirtiendo prints a logging"""

    if not file_info['parseable'] or file_info['print_count'] == 0:
        return file_info

    transformer = PrintToLoggerTransformer()

    try:
        # Transformar AST
        new_tree = transformer.visit(file_info['tree'])

        # Generar código transformado
        new_content = ast.unparse(new_tree)

        # Añadir información de transformación
        file_info.update({
            'transformed': True,
            'changes': transformer.changes,
            'new_content': new_content,
            'changes_count': len(transformer.changes)
        })

        return file_info

    except Exception as e:
        file_info.update({
            'transformed': False,
            'error': str(e)
        })
        return file_info


def find_python_files(root_dir: Path, exclude_patterns: List[str] = None) -> List[Path]:
    """Encuentra todos los archivos Python a migrar"""

    if exclude_patterns is None:
        exclude_patterns = [
            '__pycache__',
            '.git',
            '.pytest_cache',
            'venv',
            '.venv',
            'env',
            'tests',  # Excluir tests por ahora
            'config_',  # El ConfigManager ya está bien
        ]

    python_files = []

    for py_file in root_dir.rglob('*.py'):
        # Verificar si está en directorio excluido
        should_exclude = False
        for pattern in exclude_patterns:
            if pattern in str(py_file):
                should_exclude = True
                break

        if not should_exclude:
            python_files.append(py_file)

    return python_files


def main():
    parser = argparse.ArgumentParser(description='Migra print() statements a logging estructurado')
    parser.add_argument('--root-dir', type=str, default='.',
                        help='Directorio raíz del proyecto')
    parser.add_argument('--dry-run', action='store_true',
                        help='Solo analiza, no modifica archivos')
    parser.add_argument('--backup', action='store_true', default=True,
                        help='Crear backup de archivos modificados')
    parser.add_argument('--include-tests', action='store_true',
                        help='Incluir archivos de tests')

    args = parser.parse_args()

    root_path = Path(args.root_dir)

    print(f"Analizando archivos Python en: {root_path}")

    # Configurar exclusiones
    exclude_patterns = [
        '__pycache__',
        '.git',
        '.pytest_cache',
        'venv',
        '.venv',
        'env',
        'config_',  # Ya tiene logging estructurado
    ]

    if not args.include_tests:
        exclude_patterns.append('tests')

    # Encontrar archivos
    python_files = find_python_files(root_path, exclude_patterns)
    print(f"Encontrados {len(python_files)} archivos Python")

    # Analizar archivos
    print("\nAnalizando archivos...")
    file_analyses = []
    total_prints = 0

    for py_file in python_files:
        analysis = analyze_python_file(py_file)
        file_analyses.append(analysis)

        if analysis['parseable']:
            total_prints += analysis['print_count']
            if analysis['print_count'] > 0:
                print(f"  {py_file.name}: {analysis['print_count']} prints")

    print(f"\nTotal de print() statements encontrados: {total_prints}")

    if total_prints == 0:
        print("No hay print() statements para migrar")
        return

    if args.dry_run:
        print("Modo dry-run: No se modificarán archivos")
        return

    # Confirmar migración
    response = input(f"\n¿Proceder con la migración de {total_prints} print() statements? (y/N): ")
    if response.lower() not in ['y', 'yes', 'sí', 'si']:
        print("Migración cancelada")
        return

    # Transformar archivos
    print("\nTransformando archivos...")
    transformed_count = 0
    error_count = 0

    for file_info in file_analyses:
        if file_info['print_count'] > 0:
            # Crear backup si se solicita
            if args.backup:
                backup_path = file_info['file'].with_suffix('.py.backup')
                backup_path.write_text(file_info['content'], encoding='utf-8')

            # Transformar
            transformed_info = transform_file(file_info)

            if transformed_info.get('transformed', False):
                # Escribir archivo transformado
                transformed_info['file'].write_text(
                    transformed_info['new_content'],
                    encoding='utf-8'
                )

                print(f"  ✓ {transformed_info['file'].name}: "
                      f"{transformed_info['changes_count']} cambios")
                transformed_count += 1
            else:
                print(f"  ✗ {file_info['file'].name}: "
                      f"Error - {transformed_info.get('error', 'Unknown')}")
                error_count += 1

    print(f"\nMigración completada:")
    print(f"  - Archivos transformados: {transformed_count}")
    print(f"  - Errores: {error_count}")

    if transformed_count > 0:
        print(f"\nRecuerda:")
        print("1. Revisar los archivos transformados")
        print("2. Ejecutar tests para verificar que todo funciona")
        print("3. Importar 'from utils.logging_enhanced import get_logger' en módulos transformados")
        print("4. Inicializar logger con: logger = get_logger(__name__)")


if __name__ == "__main__":
    main()