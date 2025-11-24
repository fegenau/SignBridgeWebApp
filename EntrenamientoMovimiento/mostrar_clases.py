"""
Script para mostrar todas las clases disponibles para recolección
"""

# Números (10 dígitos)
NUMEROS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Abecedario (26 letras)
LETRAS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]

# Palabras/frases comunes (32 expresiones)
PALABRAS = [
    "Por_que", "Quien", "Como", "Cuando", "Donde", "Cuantos", "Que_quieres", 
    "No_lo_se", "Si", "No", "Tal_vez", "No_lo_recuerdo", "Repite_por_favor",
    "A_la_derecha", "A_la_izquierda", "En_la_entrada", "Al_final_del_pasillo",
    "En_el_segundo_piso", "En_el_edificio", "Por_las_escaleras", "Por_el_ascensor",
    "Hola", "Adios", "Como_te_llamas", "Permiso", "Nos_vemos", "Mi_casa",
    "Como_estas", "Gracias", "Por_favor", "Cuidate", "Mi_nombre"
]

def main():
    print("=== CLASES DISPONIBLES PARA RECOLECCIÓN ===\n")
    
    print("� NÚMEROS (10 dígitos):")
    print(" ".join(NUMEROS))
    print(f"Total números: {len(NUMEROS)}\n")
    
    print("�📝 ABECEDARIO (26 letras):")
    print(" ".join(LETRAS))
    print(f"Total letras: {len(LETRAS)}\n")
    
    print("💬 PALABRAS/FRASES COMUNES (32 expresiones):")
    for i, palabra in enumerate(PALABRAS, 1):
        print(f"{i:2d}. {palabra.replace('_', ' ')}")
    print(f"Total palabras/frases: {len(PALABRAS)}\n")
    
    total_clases = len(NUMEROS) + len(LETRAS) + len(PALABRAS)
    print("📊 RESUMEN:")
    print(f"Total clases: {total_clases} ({len(NUMEROS)} números + {len(LETRAS)} letras + {len(PALABRAS)} palabras/frases)")
    print(f"Muestras por clase: 30")
    print(f"Total secuencias: {total_clases * 30}")
    print(f"Tiempo estimado: ~{total_clases * 2} minutos (2 min/clase)")
    
    print("\n🚀 COMANDOS DE EJEMPLO:")
    print("# Recolectar todo:")
    print("python recoleccion.py")
    print("\n# Solo números:")
    print("python recoleccion.py --clases", " ".join(NUMEROS))
    print("\n# Solo abecedario:")
    print("python recoleccion.py --clases", " ".join(LETRAS[:5]), "...")
    print("\n# Solo preguntas básicas:")
    print("python recoleccion.py --clases Por_que Quien Como Cuando Donde")
    print("\n# Solo direcciones:")
    print("python recoleccion.py --clases A_la_derecha A_la_izquierda En_la_entrada")

if __name__ == "__main__":
    main()