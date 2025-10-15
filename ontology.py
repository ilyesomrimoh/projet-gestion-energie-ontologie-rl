from rdflib import Graph, Literal, RDF, Namespace

def create_ontology():
    # namespace c'est une adresse unique pour identifier de manière unique les eléments de l'ontologie
    EX = Namespace("http://example.org/energy#")
    g = Graph()

    g.add((EX.BatteryA, RDF.type, EX.Battery))
    g.add((EX.BatteryB, RDF.type, EX.Battery))
    g.add((EX.Motor, RDF.type, EX.Component))

    g.add((EX.BatteryA, EX.hasCharge, Literal(80)))
    g.add((EX.BatteryB, EX.hasCharge, Literal(40)))
    g.add((EX.Motor, EX.powerDemand, Literal(50)))

    g.serialize("data/ontology.ttl", format="turtle")
    return g

if __name__ == "__main__":
    create_ontology()
