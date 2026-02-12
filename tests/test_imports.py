"""
Script de prueba para verificar que todos los módulos se importan correctamente
"""

print("🔍 Verificando imports de módulos...")

try:
    print("\n✓ Importando config...")
    from config import settings
    print(f"  - PAGE_TITLE: {settings.PAGE_TITLE}")
    print(f"  - DEFAULT_K_MIN: {settings.DEFAULT_K_MIN}")
    
    print("\n✓ Importando styles...")
    from styles import apply_custom_styles, CUSTOM_CSS
    print(f"  - CSS length: {len(CUSTOM_CSS)} caracteres")
    
    print("\n✓ Importando core.data_loader...")
    from core.data_loader import load_data
    print(f"  - load_data: {load_data.__name__}")
    
    print("\n✓ Importando core.data_cleaner...")
    from core.data_cleaner import analyze_data_quality, clean_data
    print(f"  - analyze_data_quality: {analyze_data_quality.__name__}")
    print(f"  - clean_data: {clean_data.__name__}")
    
    print("\n✓ Importando core.scaler...")
    from core.scaler import scale_data
    print(f"  - scale_data: {scale_data.__name__}")
    
    print("\n✓ Importando core.clustering...")
    from core.clustering import determine_optimal_k, perform_clustering, select_best_method
    print(f"  - determine_optimal_k: {determine_optimal_k.__name__}")
    print(f"  - perform_clustering: {perform_clustering.__name__}")
    print(f"  - select_best_method: {select_best_method.__name__}")
    
    print("\n✓ Importando utils.stats...")
    from utils.stats import (
        calculate_skewness_kurtosis,
        detect_outliers_iqr,
        get_correlation_pairs,
        calculate_variance_stats
    )
    print(f"  - calculate_skewness_kurtosis: {calculate_skewness_kurtosis.__name__}")
    print(f"  - detect_outliers_iqr: {detect_outliers_iqr.__name__}")
    print(f"  - get_correlation_pairs: {get_correlation_pairs.__name__}")
    print(f"  - calculate_variance_stats: {calculate_variance_stats.__name__}")
    
    print("\n" + "="*60)
    print("✅ TODOS LOS MÓDULOS SE IMPORTARON CORRECTAMENTE")
    print("="*60)
    print("\n📊 Resumen:")
    print("  - config: ✓")
    print("  - styles: ✓")
    print("  - core: ✓ (4 módulos)")
    print("  - utils: ✓ (1 módulo)")
    print("\n🚀 La aplicación está lista para ejecutarse!")
    print("   Ejecuta: streamlit run main.py")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
