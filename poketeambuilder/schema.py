import graphene
import vg_api.schema


class Query(vg_api.schema.Query, graphene.ObjectType):
    pass


schema = graphene.Schema(query=Query)
