# settings.py - Configuracion unificada para PhiBot (EN LA RAÍZ)
"""
Sistema de configuracion modernizado que usa YAML centralizado.
Este archivo mantiene compatibilidad con el codigo existente.
"""

from config.config_manager import S, get_config, get_config_manager


def reload_config():
    """Recarga configuración desde archivo YAML"""
    return get_config_manager().load_config(force_reload=True)


def get_current_config():
    """Obtiene configuración actual completa"""
    return get_config()


def validate_current_config():
    """Valida configuración actual"""
    config_mgr = get_config_manager()
    config = config_mgr.get_config()
    errors = config_mgr.validate_config(config)

    if errors:
        print("Errores de validacion encontrados:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("Configuracion valida")
        return True


if __name__ == "__main__":
    print("Test de configuracion:")
    print(f"Data path: {S.data_path}")
    print(f"TP multiplier: {S.tp_multiplier}")
    print(f"Capital per trade: {S.capital_per_trade}")

    validate_current_config()